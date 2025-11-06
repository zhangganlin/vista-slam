from copy import copy, deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .losses_pcl import Sum, LLoss, L21, Criterion, MultiLoss, ConfLoss
from ..utils.geometry import normalize_pointcloud

from typing import Literal

class ReprojLoss (Criterion, MultiLoss):

    def __init__(self, criterion):
        super().__init__(criterion)
        self.name = 'Reproj'


    def two_view_corr(self, view_src, view_tgt):
        src_pts = view_src['pts3d_cam']
        tgt_pts = view_tgt['pts3d_cam']
        K = view_tgt['camera_intrinsics'] # (B, 3, 3)
        src_mask = view_src['valid_mask']
        tgt_mask = view_tgt['valid_mask']

        relative_pose = torch.bmm(view_tgt["camera_pose"].inverse(),view_src["camera_pose"])

        B, H, W, _ = src_pts.shape
        src_pts_hom = torch.cat([src_pts, torch.ones(B, H, W, 1, device=src_pts.device)], dim=-1)  # (B, H, W, 4)   
        src_pts_hom = src_pts_hom.view(B, -1, 4)  # (B, H*W, 4)
        src_pts_in_tgt = torch.bmm(relative_pose, src_pts_hom.transpose(1, 2)).transpose(1, 2)
        src_pts_in_tgt = src_pts_in_tgt[..., :3] / src_pts_in_tgt[..., 3:] # (B, H*W, 3)
        
        src_pts_proj = torch.bmm(K, src_pts_in_tgt.transpose(1, 2)).transpose(1, 2)
        src_pts_proj = src_pts_proj[..., :2] / src_pts_proj[..., 2:]  # (B, H*W, 2)
        src_pts_proj = src_pts_proj.view(B, H, W, 2)  # (B, H, W, 2)
        x_coords = src_pts_proj[..., 0]
        y_coords = src_pts_proj[..., 1]
        x_coords = 2 * (x_coords / (W - 1)) - 1
        y_coords = 2 * (y_coords / (H - 1)) - 1
        grid = torch.stack((x_coords, y_coords), dim=-1)  # (B, H, W, 2)
        grid = grid.view(B, H, W, 2)  # (B, H, W, 2)

        # Use grid_sample to interpolate tgt_pts_pred
        tgt_pts = tgt_pts.permute(0, 3, 1, 2)  # (B, 3, H, W)
        tgt_pts_selected = F.grid_sample(tgt_pts, grid, mode='nearest', align_corners=True)  # (B, 3, H, W)
        tgt_pts_selected = tgt_pts_selected.permute(0, 2, 3, 1)  # (B, H, W, 3)

        # Use grid_sample to interpolate tgt_mask_gt
        tgt_mask_gt = tgt_mask.unsqueeze(1).float()  # (B, 1, H, W)
        tgt_mask_selected = F.grid_sample(tgt_mask_gt, grid, mode='nearest', align_corners=True)  # (B, 1, H, W)
        tgt_mask_selected = tgt_mask_selected.squeeze(1)
        tgt_mask_selected = tgt_mask_selected >= 1.0  # (B, H, W)



        src_pts_in_tgt = src_pts_in_tgt.view(B, H, W, 3)  # (B, H, W, 3)
        visibile_mask = torch.abs((src_pts_in_tgt-tgt_pts_selected)[:,:,:,-1]) < 0.05        

        src_valid_mask = src_mask & tgt_mask_selected & visibile_mask
        return grid, src_valid_mask

        
    def proj_pts(self, view_src_pred):

        src_pts_pred = view_src_pred['pts3d_pred']  # (B, H, W, 3)

        src_to_tgt_unscale = view_src_pred['relative_pose'] # (B, 4, 4) Sim3, translation is norm to 1

        src_to_tgt = src_to_tgt_unscale.clone()
        src_to_tgt[:,:3,:3] = src_to_tgt_unscale[:,:3,:3]

        # Convert src_pts_pred to homogeneous coordinates
        B, H, W, _ = src_pts_pred.shape
        src_pts_pred_hom = torch.cat([src_pts_pred, torch.ones(B, H, W, 1, device=src_pts_pred.device)], dim=-1)  # (B, H, W, 4)   
        src_pts_pred_hom = src_pts_pred_hom.view(B, -1, 4)  # (B, H*W, 4)
        src_pts_pred_in_tgt = torch.bmm(src_to_tgt, src_pts_pred_hom.transpose(1, 2)).transpose(1, 2)
        src_pts_pred_in_tgt = src_pts_pred_in_tgt[..., :3] / src_pts_pred_in_tgt[..., 3:] # (B, H*W, 3)
        src_pts_pred_in_tgt = src_pts_pred_in_tgt.view(B, H, W, 3)  # (B, H, W, 3)

        return src_pts_pred_in_tgt




        
    def compute_loss(self, gt_views, pred_views):
        '''
        gt_views : list of dictionaries, each containing 'pts3d' and 'valid_mask'
        pred_views : list of dictionaries, each containing 'pts3d'
        '''
        all_l = []

        ls=[]

        details = {}

        main_view_gt = gt_views['main_view']


        for i in range(len(gt_views['support_views'])):
            nghb_view_gt = gt_views['support_views'][i]
            nghb_view_pred = pred_views['support_views'][i]
            main_view_pred = pred_views['main_views'][i]
            
            B,H,W = main_view_gt['valid_mask'].shape
            main_valid_mask = main_view_gt['valid_mask'].view(B, -1) # (B, H, W)
            main_pts_pred = main_view_pred['pts3d_pred'].view(B,-1,3)
            nghb_valid_mask = nghb_view_gt['valid_mask'].view(B,-1)
            nghb_pts_pred = nghb_view_pred['pts3d_pred'].view(B,-1,3)
        
            pred_pts = torch.cat([main_pts_pred,nghb_pts_pred],dim=1)       # B,N,3
            valid_mask = torch.cat([main_valid_mask,nghb_valid_mask],dim=1) # B,N
            pred_scales = torch.norm(pred_pts*valid_mask[...,None], dim=-1).sum(dim=1) / valid_mask.sum(dim=1)
            pred_scales = torch.nan_to_num(pred_scales, nan=1.0)

            corr_grid, src_valid_mask = self.two_view_corr(main_view_gt, nghb_view_gt)
            nghb_pts = nghb_view_pred['pts3d_pred'].permute(0, 3, 1, 2)  # (B, 3, H, W)
            nghb_pts_selected = F.grid_sample(nghb_pts, corr_grid, mode='nearest', align_corners=True)  # (B, 3, H, W)
            nghb_pts_selected = nghb_pts_selected.permute(0, 2, 3, 1)  # (B, H, W, 3)

            main_pts_in_nghb = self.proj_pts(main_view_pred)
            l1 = self.criterion( (main_pts_in_nghb/pred_scales[:,None,None,None]) [src_valid_mask], 
                                 (nghb_pts_selected/pred_scales[:,None,None,None])[src_valid_mask])
            
            if math.isfinite(float(l1.mean())):            
                ls.append(float(l1.mean()))
                all_l.append((l1,src_valid_mask))
        details["reproj_loss"] = sum(ls) / len(ls) if ls else 0
        return Sum(*all_l), details


class RelPoseLoss (MultiLoss):

    def __init__(self, w_rot=1.0, w_trans=1.0, trans_loss: Literal["angle", "l2"] = "angle", 
                 identity_constraint=False, testing=False, conf=False, conf_alpha=0.5):
        super().__init__()
        self.name = 'RelPose'
        self.w_rot = w_rot
        self.w_trans = w_trans
        self.testing = testing
        self.trans_loss_type = trans_loss
        self.identity_constraint = identity_constraint
        self.compare_gt = True
        self.use_conf = conf
        self.conf_alpha = conf_alpha
        
        if trans_loss == "angle":
            self.trans_loss = self.get_trans_err_angle_batch
        elif trans_loss == "l2":
            self.trans_loss = self.get_trans_err_l2_batch
        else:
            raise ValueError(f"trans_loss must be 'angle' or 'l2', but got {trans_loss}")


    def get_rot_err_batch(self, rot_a, rot_b):
        """
        Calculate the rotation error between two batches of rotation matrices.
        
        Args:
        rot_a (torch.Tensor): Batch of rotation matrices A of shape (B, 3, 3).
        rot_b (torch.Tensor): Batch of rotation matrices B of shape (B, 3, 3).
        
        Returns:
        torch.Tensor: Batch of rotation errors of shape (B,).
        """
        rot_err = torch.bmm(rot_a.transpose(1, 2), rot_b)  # (B, 3, 3)
        trace = torch.diagonal(rot_err, dim1=-2, dim2=-1).sum(-1)  # Sum of diagonal elements
        theta = torch.acos(torch.clamp((trace - 1) / 2, -0.99999, 0.99999))  # Clamp for stability
        
        if not torch.isfinite(theta).all():
            print("rot theta is nan",force=True)
            print(theta, force=True)
            print(rot_err, force=True)
            print(rot_a, force=True)
            print(rot_b, force=True)
        
        return theta

    def get_trans_err_angle_batch(self,trans_a,trans_b):
        # trans_a and trans_b must be normalized
        dot_product = torch.sum(trans_a * trans_b, dim=1)  # (B,)
        cos_angle = torch.clamp(dot_product/ (torch.norm(trans_a, dim=1) * torch.norm(trans_b, dim=1)), 
                                -0.99999, 0.99999) # (B,)
        theta = torch.acos(cos_angle)  # (B,)
        theta = torch.nan_to_num(theta, nan=0.0)       # some sample's gt, trans vector is 0, thus cos_angle is nan

        if not torch.isfinite(theta).all():
            print("trans theta is nan",force=True)
            print(theta, force=True)
            print(cos_angle, force=True)
            print(trans_a, force=True)
            print(trans_b, force=True)

        return theta

    def get_trans_err_l2_batch(self,trans_a,trans_b):
        # trans_a and trans_b must be normalized
        error = torch.nan_to_num(trans_a - trans_b, nan=0.0)
        err = torch.norm(error, p=2, dim=1)
        return err
    
    def get_trans_length_batch(self,trans_a,trans_b):
        len_a = torch.norm(trans_a, dim=1)  # (B,)
        len_b = torch.norm(trans_b, dim=1)
        loss = F.l1_loss(len_a, len_b)
        return loss
        
        
    def compute_loss(self, gt_views, pred_views, debug=False):
        '''
        gt_views : list of dictionaries, each containing 'pts3d' and 'valid_mask'
        pred_views : list of dictionaries, each containing 'pts3d'

        IMPORTANT: the translation in relative pose should be normalized

        '''
        assert self.compare_gt or self.identity_constraint
        
        total_loss = 0

        ls_rot_relative=[]
        ls_trans_relative=[]
        ls_trans_length=[]

        ls_rot_id = []
        ls_trans_id = []

        ls_conf_loss = []
        details = {}

        main_view_gt = gt_views['main_view']
        main_pcl_gt = main_view_gt['pts3d_cam'] # (B, H, W, 3)
        main_valid = main_view_gt['valid_mask'] # (B, H, W)
        
        for i in range(len(gt_views['support_views'])):
            nghb_view_gt = gt_views['support_views'][i]
            nghb_view_pred = pred_views['support_views'][i]
            main_view_pred = pred_views['main_views'][i]

            main_pcl_pred = main_view_pred['pts3d_pred']
            nghb_pcl_pred = nghb_view_pred['pts3d_pred']

            nghb_pcl_gt = nghb_view_gt['pts3d_cam'] # (B, H, W, 3)
            nghb_valid = nghb_view_gt['valid_mask'] # (B, H, W)

            main_to_nghb_pred = main_view_pred['relative_pose']
            pose_conf = main_view_pred['relative_pose_conf']
            nghb_to_main_pred = nghb_view_pred['relative_pose']


            gt_norm_factor = normalize_pointcloud(main_pcl_gt,nghb_pcl_gt,'avg_dis',
                                                  main_valid,nghb_valid,return_factor_only=True).squeeze(-1).squeeze(-1)
            pred_norm_factor = normalize_pointcloud(main_pcl_pred,nghb_pcl_pred,'avg_dis',
                                                  main_valid,nghb_valid,return_factor_only=True).squeeze(-1).squeeze(-1)

            main_to_nghb_gt = torch.bmm(nghb_view_gt["camera_pose"].inverse(),main_view_gt["camera_pose"])
            main_to_nghb_gt_rot = main_to_nghb_gt[:,:3,:3]
            main_to_nghb_gt_trans = main_to_nghb_gt[:,:3,3] / gt_norm_factor
            
            main_to_nghb_pred_rot = main_to_nghb_pred[:,:3,:3]
            main_to_nghb_pred_trans = main_to_nghb_pred[:,:3,3] / pred_norm_factor

            nghb_to_main_pred_rot = nghb_to_main_pred[:,:3,:3]
            nghb_to_main_pred_trans = nghb_to_main_pred[:,:3,3] / pred_norm_factor


            if not torch.isfinite(main_to_nghb_pred).all() or not torch.isfinite(nghb_to_main_pred).all():
                print("pred relative pose is nan",force=True)
                print(main_to_nghb_pred, force=True)
                print(nghb_to_main_pred, force=True)
            

            rot_err = self.get_rot_err_batch(main_to_nghb_pred_rot,main_to_nghb_gt_rot)

            trans_err = self.trans_loss(main_to_nghb_pred_trans,main_to_nghb_gt_trans)

            if self.testing:
                trans_length_err = self.get_trans_length_batch(main_to_nghb_pred_trans,main_to_nghb_gt_trans)

            if self.identity_constraint:
                rot_together = torch.bmm(main_to_nghb_pred_rot,nghb_to_main_pred_rot)
                identity_rot = torch.eye(3, device=rot_together.device).unsqueeze(0).expand(rot_together.shape[0], -1, -1)
                rot_err_2 = self.get_rot_err_batch(rot_together,identity_rot)

                rel_trans_a = main_to_nghb_pred_trans
                rel_trans_b = torch.bmm(main_to_nghb_pred_rot,
                                        nghb_to_main_pred_trans.unsqueeze(-1)).squeeze(-1)

                if self.trans_loss_type == "angle":
                    trans_err_2 = self.get_trans_err_angle_batch(rel_trans_a,-rel_trans_b)
                elif self.trans_loss_type == "l2":
                    trans_err_2 = self.get_trans_err_l2_batch(rel_trans_a, -rel_trans_b)

            rot_err_sum = 0
            trans_err_sum = 0
            
            if self.compare_gt:
                rot_err_sum += torch.abs(rot_err)
                trans_err_sum += torch.abs(trans_err)
            if self.identity_constraint:
                rot_err_sum += rot_err_2
                trans_err_sum += trans_err_2
            
            
            if self.use_conf:
                weighted_loss = (self.w_rot * rot_err_sum + self.w_trans * trans_err_sum) * pose_conf
                total_loss += (weighted_loss - self.conf_alpha*torch.log(pose_conf)).sum()
            else:
                total_loss += self.w_rot * rot_err_sum.sum() + self.w_trans * trans_err_sum.sum()


            if self.compare_gt:
                ls_rot_relative.append(float(rot_err.mean()))
                ls_trans_relative.append(float(trans_err.mean()))
                if self.testing:
                    ls_trans_length.append(float(trans_length_err.mean()))
            if self.identity_constraint:
                ls_rot_id.append(float(rot_err_2.mean()))
                ls_trans_id.append(float(trans_err_2.mean()))
            if self.use_conf:
                ls_conf_loss.append(float(total_loss.mean()))

        if self.compare_gt:
            details["rot_loss"] = sum(ls_rot_relative) / len(ls_rot_relative) if ls_rot_relative else 0
            details[f"trans_loss_{self.trans_loss_type}"] = sum(ls_trans_relative) / len(ls_trans_relative) if ls_trans_relative else 0
            if self.testing:
                details["trans_loss_length"] = sum(ls_trans_length) / len(ls_trans_length) if ls_trans_length else 0
        if self.identity_constraint:
            details["rot_identity_loss"] = sum(ls_rot_id) / len(ls_rot_id) if ls_rot_id else 0
            details[f"trans_identity_loss_{self.trans_loss_type}"] = sum(ls_trans_id) / len(ls_trans_id) if ls_trans_id else 0
        if self.use_conf:
            details["pose_conf_loss"] = sum(ls_conf_loss) / len(ls_conf_loss) if ls_conf_loss else 0

        return total_loss, details

