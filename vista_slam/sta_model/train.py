import os
import sys
import math
import json
import time
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from typing import Sized
from shutil import copyfile
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils import croco_misc as misc 
from ..datasets import build_dataset
from ..utils.misc import backup_src_dir
from ..utils.croco_misc import NativeScalerWithGradNormCount as NativeScaler
from ..utils.croco_misc import get_grad_norm_
from .losses_pcl import ConfLoss, PointRegrLoss, MultiLoss, L21
from .losses_geo import RelPoseLoss, ReprojLoss
from .sta_model import SymmetricTwoViewAssociation

import getpass


output_base = None
pretrained = "pretrain.pth"

scannet_sensor_data_root='/datasets/scannet/scans'
scannet_view_graph_root = '/datasets/viewgraph_scannet'

scannetpp_sensor_data_root='/datasets/scannetpp'
scannetpp_view_graph_root = '/datasets/viewgraph_scannetpp'

sevenscenes_sensor_data_root='/datasets/7scenes'
sevenscenes_view_graph_root = '/datasets/viewgraph_7scenes'

arkit_sensor_data_root='/datasets/arkitscenes'
arkit_view_graph_root = '/datasets/viewgraph_arkit'

co3d_sensor_data_root='/datasets/co3d'

ase_sensor_data_root='/datasets/ase'
ase_view_graph_root = '/datasets/viewgraph_ase'

replica_data_root_b='/datasets/replica_rendering_b'
replica_data_root_c='/datasets/replica_rendering_c'
replica_data_root_d='/datasets/replica_rendering_d'



###############dataset setting################
para_neighbor_num=1
para_loop_num=2
batch_size=16

support_num = para_neighbor_num*2 + para_loop_num

# para_neighbor_num=3
# para_loop_num=3
# batch_size=10

def get_args_parser():
    parser = argparse.ArgumentParser('SymmetricTwoViewAssociation training', add_help=False)
    parser.add_argument('--model', default="SymmetricTwoViewAssociation(freeze='none',conf_mode=('exp', 0, 1))",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=pretrained, help='path of a starting checkpoint')
    parser.add_argument('--resume', action='store_true', help='whether to resume existed run')
    parser.add_argument('--print_info', action='store_true', help='whether print model, optimizer and param_group')
    

    parser.add_argument('--train_dataset', 
                        default=f"10000@ScanNet(split='train', resolution=(224,224), \
                                         sensor_data_root='{scannet_sensor_data_root}', \
                                         view_graph_root='{scannet_view_graph_root}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  10000@ScanNetpp(split='train', resolution=(224,224), \
                                         sensor_data_root='{scannetpp_sensor_data_root}', \
                                         view_graph_root='{scannetpp_view_graph_root}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  10000@ARKitScene(split='train', resolution=(224,224), \
                                         sensor_data_root='{arkit_sensor_data_root}', \
                                         view_graph_root='{arkit_view_graph_root}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  10000@Co3d(split='train', resolution=(224,224), \
                                         sensor_data_root='{co3d_sensor_data_root}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  10000@AriaSynthetic(split='train', resolution=(224,224), \
                                         sensor_data_root='{ase_sensor_data_root}', \
                                         view_graph_root='{ase_view_graph_root}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  3000@Replica(split='train', resolution=(224,224), \
                                         sensor_data_root='{replica_data_root_b}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  3000@Replica(split='train', resolution=(224,224), \
                                         sensor_data_root='{replica_data_root_c}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)+ \
                                  3000@Replica(split='train', resolution=(224,224), \
                                         sensor_data_root='{replica_data_root_d}', \
                                         neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                         scene_name=None)\
                                ",
                        type=str, help="training dataset")
    

    parser.add_argument('--test_dataset', 
                        default=f"SevenScenes(split='test', resolution=(224,224), \
                                    sensor_data_root='{sevenscenes_sensor_data_root}', \
                                    view_graph_root='{sevenscenes_view_graph_root}',\
                                    neighbor_num={para_neighbor_num}, loop_num={para_loop_num}, \
                                    scene_name=None)",
                        type=str, help="testing dataset")

    # Loss

    parser.add_argument('--train_criterion', 
                        default="ConfLoss(PointRegrLoss(L21,mode='train'),alpha=0.4) + RelPoseLoss(trans_loss='l2',identity_constraint=True, conf=True, conf_alpha=0.05) + ReprojLoss(L21)",
                        type=str, help="train criterion")    

    parser.add_argument('--test_criterion', 
                        default="ConfLoss(PointRegrLoss(L21,mode='test'),alpha=0.1) + RelPoseLoss(trans_loss='angle',identity_constraint=True,testing=True) + ConfLoss(ReprojLoss(L21),alpha=0.1)",
                        type=str, help="test criterion")
    
     # Exp
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    
    # Training
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=200, type=int, help="Maximum number of epochs for the scheduler")
    
    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=1.5e-5, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-06, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    
    # others
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=0, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')
    
    parser.add_argument('--alpha_c2f', type=int, default=1, help='use alpha c2f')
    
    # output dir 
    parser.add_argument('--output', default=f'output/test', type=str, help="path where to save the output")
    


    return parser


@torch.no_grad()
def test_one_epoch(model: SymmetricTwoViewAssociation, criterion: MultiLoss,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    
    save_path = os.path.join(args.output, f'test')
        
    os.makedirs(save_path, exist_ok=True)

    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        for view in [batch['main_view']]+batch['neighbor_views']+batch['loop_views']:
            for name in 'img pts3d_cam valid_mask camera_pose camera_intrinsics'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
        
        support_num_to_use = support_num     
        pre_views = model.forward(batch, support_num_to_use)
        support_views = batch['neighbor_views']+batch['loop_views']
        support_views = support_views[:support_num_to_use]
        gt_views = {"main_view":batch['main_view'], "support_views":support_views}

        loss, loss_details = criterion(gt_views, pre_views)
        loss_value = float(loss)
        
        metric_logger.update(loss=float(loss_value), **loss_details)
        # metric_logger.update(pseudo_v_loss=float(loss_mem_log))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix+'_'+name, val, 1000*epoch)

    return results


def train_one_epoch(model: SymmetricTwoViewAssociation, criterion: MultiLoss,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler:NativeScaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
        
    epoch_ratio = epoch/args.epochs
    if epoch_ratio < 0.75:
        active_ratio = min(1, epoch/args.epochs*2.0)
    else:
        active_ratio = max(0.5, 1 - (epoch_ratio - 0.75) / 0.25)
    
    
    optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)
        

        for view in [batch['main_view']]+batch['neighbor_views']+batch['loop_views']:
            for name in 'img pts3d_cam valid_mask camera_pose camera_intrinsics'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
                
        support_num_to_use = support_num     
        pre_views = model.forward(batch, support_num_to_use)

        support_views = batch['neighbor_views']+batch['loop_views']
        support_views = support_views[:support_num_to_use]
        gt_views = {"main_view":batch['main_view'], "support_views":support_views}

        loss, loss_details = criterion(gt_views, pre_views)
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            print(loss_details, force=True)
            sys.exit(1)

        loss /= accum_iter
        norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0, clip_grad=1.0) # 
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)
        # metric_logger.update(pseudo_v_loss=float(loss_mem_log))

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            # pseudo_v_loss_reduce = misc.all_reduce_mean(float(loss_mem_log))
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('train_pseudo_v_loss', pseudo_v_loss_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            log_writer.add_scalar('active_ratio', active_ratio, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_'+name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    args.output = os.path.join(output_base, args.output)

    print("output_dir: "+args.output)
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # auto resume
    last_ckpt_fname = os.path.join(args.output, f'checkpoint-last.pth')
    
    if args.resume and os.path.isfile(last_ckpt_fname):
        args.resume = last_ckpt_fname
    else:
        args.resume = None
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)
    data_loader_test = build_dataset(args.test_dataset, args.batch_size, args.num_workers, test=True)
    
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    
    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion:MultiLoss = eval(args.train_criterion).to(device)
    test_criterion:MultiLoss = eval(args.test_criterion).to(device)

    alpha_init = train_criterion.alpha
    
    model.to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.LayerNorm):
            m.float()
    model_without_ddp = model
    if args.print_info: print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        if "model" in ckpt.keys():
            ckpt = ckpt["model"]
        print(model.load_state_dict(ckpt, strict=False))
        del ckpt  # in case it occupies memory
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=False)
        model_without_ddp = model.module
        
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay, print_info=args.print_info)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    if args.print_info: print(optimizer)
    
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()},
                                          **{f'test_{k}': v for k, v in test_stats.items()})

            with open(os.path.join(args.output, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)
    
    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output is not None:
        log_writer = SummaryWriter(log_dir=args.output)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs+1):
        
        # TODO: Save last check point
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch-1, 'last', best_so_far)
        
        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = test_one_epoch(model, test_criterion, data_loader_test,
                                    device, epoch, log_writer=log_writer, args=args, prefix="test")
            # Save best of all
            if test_stats['loss_med'] < best_so_far:
                best_so_far = test_stats['loss_med']
                new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)
        
        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch-1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch-1, 'best', best_so_far)
            
        if epoch >= args.epochs:
            break 

        if args.alpha_c2f:
            train_criterion.alpha = alpha_init - 0.2 * max((epoch - 0.5 * args.epochs) / (0.5 * args.epochs), 0)
            print('Update alpha to', train_criterion.alpha)
        
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    