# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy
import torch
import torch.nn as nn

from ..utils.geometry import normalize_pointcloud

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]

    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        error = torch.nan_to_num(a - b, nan=0.0)
        return torch.norm(error, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details

class PointRegrLoss (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
        gt and pred are transformed into localframe1
    """

    def __init__(self, criterion, mode="train"):
        super().__init__(criterion)
        self.name="PointRegr"
        self.mode=mode
        
    def get_scale_main_view(self, gt_pcds, pred_pcds, valid_masks):
        """
        gt_pcds: (B, H, W, 3)
        pred_pcds: (B, H, W, 3)
        valid_masks: (B, H, W)
        Align the scale of the predicted point clouds to the ground truth point clouds.
        """
        # Mask invalid points
        valid_gt_pcds = gt_pcds * valid_masks.unsqueeze(-1)
        valid_pred_pcds = pred_pcds * valid_masks.unsqueeze(-1)
        
        # Compute scales
        gt_scales = torch.norm(valid_gt_pcds, dim=-1).sum(dim=[1, 2]) / valid_masks.sum(dim=[1, 2])
        pred_scales = torch.norm(valid_pred_pcds, dim=-1).sum(dim=[1, 2]) / valid_masks.sum(dim=[1, 2])
        # pred_scales = 1.0
        # Compute scale factors
        scale_factors = gt_scales / pred_scales

        if torch.isnan(scale_factors).any():
            # Handle NaN values\
            print(f"NaN values detected in scale factors: {scale_factors}", force=True)
            print(f"gt_scales: {gt_scales}", force=True)
            print(f"pred_scales: {pred_scales}", force=True)
            print(f"valid_gt_pcds: {torch.norm(valid_gt_pcds, dim=-1).sum(dim=[1, 2]) }", force=True)
            print(f"valid_masks: {valid_masks.sum(dim=[1,2])}", force=True)
                        
            scale_factors = torch.where(torch.isnan(scale_factors), torch.ones_like(scale_factors), scale_factors)
        
        return scale_factors
        
        # # Align predicted point clouds
        # aligned_pred_pcds = pred_pcds * scale_factors.view(-1, 1, 1, 1)
        
        # return aligned_pred_pcds  

    def compute_loss(self, gt_views, pred_views):
        '''
        gt_views : list of dictionaries, each containing 'pts3d' and 'valid_mask'
        pred_views : list of dictionaries, each containing 'pts3d'
        '''
        all_l = []
        details = {}
        total_loss = []

        gt_pts_main = gt_views["main_view"]['pts3d_cam']
        valid_mask_main = gt_views["main_view"]['valid_mask']

        for i in range(len(pred_views["main_views"])):
            gt_pts_supp = gt_views['support_views'][i]['pts3d_cam']
            valid_mask_supp = gt_views["support_views"][i]['valid_mask']

            pred_pts_main = pred_views["main_views"][i]['pts3d_pred']
            pred_pts_supp = pred_views["support_views"][i]['pts3d_pred']

            gt_res = normalize_pointcloud(gt_pts_main,gt_pts_supp,'avg_dis',valid_mask_main,valid_mask_supp)
            gt_main_norm, gt_supp_norm = gt_res[0], gt_res[1]
            pred_res = normalize_pointcloud(pred_pts_main,pred_pts_supp,'avg_dis',valid_mask_main,valid_mask_supp)
            pred_main_norm, pred_supp_norm = pred_res[0], pred_res[1]

            l1_main = self.criterion(pred_main_norm[valid_mask_main], gt_main_norm[valid_mask_main])
            l1_supp = self.criterion(pred_supp_norm[valid_mask_supp], gt_supp_norm[valid_mask_supp])
            all_l += [(l1_main,valid_mask_main), (l1_supp,valid_mask_supp)]

            total_loss += [float(l1_main.mean()) , float(l1_supp.mean())]
        details[self.name+f'_pts3d_main'] = sum(total_loss) / len(total_loss) if total_loss else 0
        return Sum(*all_l), details


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds):
            # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds)
        res_loss = 0
        res_info = details
        pts_loss = []

        if self.pixel_loss.name == "PointRegr":
            for j in range(len(gts["support_views"])):
                for i, conf in [(j*2,   preds['main_views'][j]['conf']), 
                                (j*2+1, preds['support_views'][j]['conf'])]:
                    loss = losses_and_masks[i][0]
                    mask = losses_and_masks[i][1]
                    conf, log_conf = self.get_conf_log(conf[mask])
                    conf_loss = loss * conf - self.alpha * log_conf
                    conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0                
                    res_loss += conf_loss

                    pts_loss.append(float(conf_loss))
            res_info[f'conf_loss_pts3d_main'] = sum(pts_loss) / len(pts_loss) if pts_loss else 0

        elif self.pixel_loss.name == "Reproj":
            for i in range(len(losses_and_masks)):
                loss = losses_and_masks[i][0]
                mask = losses_and_masks[i][1]
                conf, log_conf = self.get_conf_log(preds['main_views'][i]['conf'][mask])
                conf_loss = loss * conf - self.alpha * log_conf
                conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0    

                res_loss += conf_loss
            res_info[f'conf_loss_reproj'] = float(res_loss) / len(losses_and_masks)    
        
        return res_loss, res_info


