import torch
import numpy as np
from vista_slam.utils.slam_utils import compute_local_pointclouds, FontColor, print_msg
from colorama import Fore
from vista_slam.slam import OnlineSLAM
import torch.backends.cudnn as cudnn
import rerun as rr
from vista_slam.datasets.slam_images_only import SLAM_image_only
import glob, time, argparse, yaml, munch, os, tqdm

print=print_msg

def log_view(topic, cam_pose, img, pts3d, K, pts3d_mask,show_img=True, show_camera=True, pointmap=None, pointcloud=True):
    img_shape = img.shape[1:3]

    if K is None:
        K = np.eye(3, dtype=np.float32)
        K[0,0] = img_shape[1] / 2.0
        K[1,1] = img_shape[0] / 2.0
        K[0,2] = img_shape[1] / 2.0
        K[1,2] = img_shape[0] / 2.0

    img = (img+1.0)/2.0

    pose = cam_pose
                
    rr.log(f"world/est/{topic}",rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]))
    if show_camera:
        # rr.log(f"world/cam/{topic}",rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]))
        rr.log(f"world/est/{topic}/cam",
                rr.Pinhole(resolution=[img_shape[0], img_shape[1]],
                            image_from_camera=K, 
                            camera_xyz=rr.ViewCoordinates.RDF
                            )
                )

    img = img.numpy().transpose(1,2,0)
    pts = pts3d[pts3d_mask]
    colors = img[pts3d_mask]

    if pointcloud:
        rr.log(f"world/est/{topic}/points",rr.Points3D(pts,colors=colors,radii=0.002))
    
    if show_img and show_camera:
        img=(img*255).astype(np.uint8)
        if pointmap is not None:
            img = pointmap
        rr.log(f"world/est/{topic}/cam",rr.Image(img))


def rerun_vis_views(slam:OnlineSLAM, show_all):
    est_poses = [None for _ in range(slam.view_num)]
    local_pcls = [None for _ in range(slam.view_num)]
    masks = [None for _ in range(slam.view_num)]
    
    if show_all:
        to_show = list(range(slam.view_num))
        for v in range(slam.view_num):
            rr.log(f"world/est/cam_{v}",rr.Clear(recursive=True))
    else:
        to_show = [slam.view_num-1]

    for v in to_show:
        
        view = slam.get_view(v)
        pose, depth, intri = view.pose, view.depth, view.intri
        
        pose = pose.cpu().numpy()
        pcl = compute_local_pointclouds(depth.unsqueeze(0),intri).squeeze(0)
        pointmap,_ = slam.get_pointmap_vis(v)
        if show_all:
            est_poses[v] = pose
            local_pcls[v] = pcl
            masks[v] = pcl[:,:,2]>0
        pcl = pcl.cpu().numpy()
        pcl_mask = pcl[:,:,2]>0
        k = intri.cpu().numpy()
        show_pointcloud = True
        log_view(f"cam_{v}",pose,
                slam.imgs[v],pcl,k,pcl_mask,show_img=True, show_camera=True, pointmap=pointmap, pointcloud=show_pointcloud)
    return est_poses, local_pcls, masks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--images", type=str, required=True, help="Path to the input images, e.g. '/path/to/images/*.color.png'")
    parser.add_argument("--output", type=str, help="Path to the output folder, overrides the config file setting")
    parser.add_argument("--vis", action="store_true", help="Enable visualization, overrides the config file setting")
    parser.add_argument("--vis_save", action="store_true", help="Save visualization, overrides the config file setting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging, overrides the config file setting")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.output is not None:
        cfg['output_dir'] = args.output
    if args.vis:
        cfg['rerun_vis'] = True
    if args.vis_save:
        cfg['rerun_save'] = True
    if args.verbose:
        cfg['verbose'] = True
    cfg['images_path'] = args.images
    cfg = munch.Munch(cfg)
    
    # fix the seed
    random_seed = cfg.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = True
    
    output_folder = cfg.output_dir
    os.makedirs(output_folder, exist_ok=True)
    dataset = SLAM_image_only(glob.glob(cfg.images_path), resolution=(224,224))

    stride = cfg.stride
    slam = OnlineSLAM(
        ckpt_path=cfg.STA_pretrain_path , 
        vocab_path=cfg.vocab_path, 
        verbose=cfg.verbose, 
        max_view_num=cfg.max_view_num,
        neighbor_edge_num=cfg.neighbor_edge_num, 
        loop_edge_num=cfg.loop_edge_num,
        loop_dist_min=cfg.loop_dist_min, 
        loop_nms=cfg.loop_nms,
        loop_cand_thresh_neighbor=cfg.loop_cand_thresh_neighbor,
        conf_thres=cfg.point_conf_thres, 
        rel_pose_thres=cfg.rel_pose_thres,
        flow_thres=cfg.flow_thres, 
        pgo_every=cfg.pgo_every
    )
    
    if cfg.rerun_vis or cfg.rerun_save:
        rr.init(f"slam",spawn=False)
        if cfg.rerun_save:
            rr.save(output_folder+"/rerun_recording.rrd")
        if cfg.rerun_vis:
            rr.connect_grpc(cfg.rerun_url,flush_timeout_sec=None)
        rr.log(f"/world",rr.Transform3D())

    first = True
    last = len(dataset)

    if cfg.keyframe_detection == "stride":
        stride_idxes = list(range(1, last, stride))
        if len(stride_idxes) > cfg.max_view_num:
            print(f"Too many input keyframes ({len(stride_idxes)}), only using {cfg.max_view_num} images evenly sampled from the sequence for SLAM.",
                  color=FontColor.WARNING)
            stride_idxes =  list(torch.linspace(0, last - 1, steps=cfg.max_view_num).long())
            
    
    is_optimized = False
    
    read_data_time = 0
    read_data_start = time.time()
    using_stride_for_kf = (cfg.keyframe_detection == "stride")
    t = 0

    pbar = tqdm.tqdm(total=last, position=0, 
                     bar_format=Fore.GREEN+"[Progress] "+Fore.RESET+"{percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt} frames]")
    while t < last:
        pbar.n = int(t+1)
        pbar.refresh()
        if using_stride_for_kf:
            data = None
            is_keyframe = (t in stride_idxes)
        else:
            data = dataset[t]
            img_gray = (data.gray.squeeze(0).numpy()*255).astype(np.uint8)
            is_keyframe = slam.flow_tracker.compute_disparity(img_gray, visualize=False)

        if not is_keyframe:
            if t == last - 1 and not is_optimized:
                slam.pose_graph_optimize()
                torch.cuda.empty_cache()
                rerun_vis_views(slam,show_all=True)
            t+=1
            continue
        
        if data is None:  # if using stride keyframe detection, data is not read in this iteration
            data = dataset[t] 
            img_gray = (data.gray.squeeze(0).numpy()*255).astype(np.uint8)
                
        img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)
        img = data.rgb.unsqueeze(0).to(slam.device)
        
        input_value = {'rgb':img, 'shape':img_shape, 'gray':img_gray, 'view_name':data.img_name}
        read_data_time += time.time() - read_data_start
        
        is_optimized = slam.step(input_value, force_pgo=(t == last - 1))

        if first:
            first = False
            t+=1
            continue

        if slam.view_num > cfg.max_view_num:
            if cfg.keyframe_detection == "flow_stride":
                print(f"Max view number {cfg.max_view_num} reached, retrying with 'stride' keyframe detection strategy, with stride={stride} ...",
                      color=FontColor.WARNING)
                stride_idxes = list(range(1, last, stride))
                if len(stride_idxes) > cfg.max_view_num:
                    print(f"Too many input keyframes ({len(stride_idxes)}), only using {cfg.max_view_num} images evenly sampled from the sequence for SLAM.",
                          color=FontColor.WARNING)
                    stride_idxes =  list(torch.linspace(0, last - 1, steps=cfg.max_view_num).long())
                using_stride_for_kf = True
                first = True
                read_data_time = 0
                t = 0
                slam.reset()
                if cfg.rerun_vis or cfg.rerun_save:
                    rr.disconnect()
                    rr.init(f"slam_2nd_try",spawn=False)
                    if cfg.rerun_save:
                        rr.save(output_folder+"/rerun_recording_2nd_try.rrd")
                    if cfg.rerun_vis:
                        rr.connect_grpc(cfg.rerun_url,flush_timeout_sec=None)
                    rr.log(f"/world",rr.Transform3D())
                read_data_start = time.time()
                continue
            else:
                slam.pose_graph_optimize()
                is_optimized = True
                torch.cuda.empty_cache()
                print(f"Max view number {cfg.max_view_num} reached, stopping ViSTA-SLAM," 
                    "please increase max_view_num in the config file , or use 'flow_stride' or 'stride' option for keyframe_detection.",
                      color=FontColor.WARNING)
                break

        rr.set_time("index",sequence=t)
        rerun_vis_views(slam, is_optimized)            
        read_data_start = time.time()
        t+=1
        
    pbar.close()
    print(f"Total keyframes detected: {slam.view_num}", color=FontColor.INFO)
    
    time_dict = slam.get_time_dict()
    time_dict['prepare_data'] += read_data_time
    time_dict['total'] = time_dict['total'] + read_data_time
    print(f"Total time spent: {time_dict['total']:.1f} s", color=FontColor.INFO)
    if slam.verbose: print(f"Time spent in each step:{time_dict}", color=FontColor.INFO)
    
    rr.disconnect()

    print(f"Saving data to {output_folder} ...", color=FontColor.INFO, end=" ")
    slam.save_data_all(f"{output_folder}",
                        save_view_graph=True, traj_name_postfix=None,
                        save_poses=True, save_images=True, save_scales=True,
                        save_depths=True, save_intrinsics=True, 
                        save_confs=True, save_ply=True,
                        gt_poses=None, gt_depths=None, gt_intrinsics=None)
    print(f"Done.")

    
