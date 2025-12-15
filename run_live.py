import torch
import numpy as np
from vista_slam.utils.slam_utils import compute_local_pointclouds, FontColor, print_msg
from colorama import Fore
from vista_slam.slam import OnlineSLAM
import torch.backends.cudnn as cudnn
import rerun as rr
from vista_slam.datasets.slam_images_only import SLAM_image_only
import glob, time, argparse, yaml, munch, os, tqdm
import threading
import cv2
print=print_msg

def log_view(topic, cam_pose, img, pts3d, K, pts3d_mask,show_img=True, show_camera=True, pointmap=None, pointcloud=True, downsample=1.0):
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
    
    # 💡 Downsample the pointcloud to save memory of visualization
    N = len(pts)
    downsample = max(0.0, min(1.0, downsample))
    if downsample < 1.0:
        idx = np.random.choice(N, int(N * downsample), replace=False)
        pts = pts[idx]
        colors = colors[idx]
    
    
    if pointcloud:
        rr.log(f"world/est/{topic}/points",rr.Points3D(pts,colors=colors,radii=0.002))
    
    if show_img and show_camera:
        img=(img*255).astype(np.uint8)
        if pointmap is not None:
            img = pointmap
        rr.log(f"world/est/{topic}/cam",rr.Image(img))


def rerun_vis_views(slam:OnlineSLAM, num_to_show):
    
    to_show = list(range(max(0, slam.view_num-num_to_show), slam.view_num))

    for v in to_show:
        view = slam.get_view(v)
        pose, depth, intri = view.pose, view.depth, view.intri
        pose = pose.cpu().numpy()
        pcl = compute_local_pointclouds(depth.unsqueeze(0),intri).squeeze(0)
        pointmap,_ = slam.get_pointmap_vis(v)

        pcl = pcl.cpu().numpy()
        pcl_mask = pcl[:,:,2]>0
        k = intri.cpu().numpy()
        show_pointcloud = True

        idx = slam.view_num-1 - v
        topic = f"cam_{idx}"
        log_view(topic,pose,
                slam.imgs[v],pcl,k,pcl_mask,show_img=False, show_camera=True, pointmap=pointmap, pointcloud=show_pointcloud,
                downsample=0.2)
    return 

class LatestCamera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._reader)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            else:
                frame = self.frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--camera", type=str, required=True, help="Path to the camera path like '/dev/video0'")
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
    cfg['camera'] = args.camera
    cfg['keyframe_detection'] = 'flow'  # only flow based keyframe detection is supported in live mode currently
    cfg['pgo_every'] = 50  # set a default value for pgo_every in live mode
    
    cfg = munch.Munch(cfg)
    
    # fix the seed
    random_seed = cfg.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = True
    
    output_folder = cfg.output_dir
    os.makedirs(output_folder, exist_ok=True)
    dataset = SLAM_image_only("", resolution=(224,224))

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
        pgo_every=cfg.pgo_every,
        live_mode=True
    )
    
    if cfg.rerun_vis or cfg.rerun_save:
        rr.init(f"slam",spawn=False)
        if cfg.rerun_save:
            rr.save(output_folder+"/rerun_recording.rrd")
        if cfg.rerun_vis:
            rr.connect_grpc(cfg.rerun_url)
        rr.log(f"/world",rr.Transform3D())

    first = True
    last = cfg.max_view_num

    assert cfg.keyframe_detection == 'flow', "Only 'flow' keyframe detection is supported in live mode currently."
    
    is_optimized = False
    
    read_data_time = 0
    read_data_start = time.time()
    t = 0

    cam = LatestCamera(cfg.camera)

    while t < last:
        
        frame = None
        while frame is None:
            frame = cam.read()
        
        data = dataset.process_image(frame, f"{t:06d}")
        img_gray = (data.gray.squeeze(0).numpy()*255).astype(np.uint8)
        is_keyframe = slam.flow_tracker.compute_disparity(img_gray, visualize=False)

        if not is_keyframe:
            if t == last - 1 and not is_optimized:
                slam.pose_graph_optimize()
                torch.cuda.empty_cache()
                rerun_vis_views(slam, num_to_show=cfg.rerun_vis_view_max)
            continue
                
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
            slam.pose_graph_optimize()
            is_optimized = True
            torch.cuda.empty_cache()
            print(f"Max view number {cfg.max_view_num} reached, stopping ViSTA-SLAM," 
                "please increase max_view_num in the config file.",
                    color=FontColor.WARNING)
            break

        # rr.set_time("index",sequence=t)
        rerun_vis_views(slam, is_optimized)            
        read_data_start = time.time()
        t+=1
        
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
    cam.stop()
    print(f"Done.")

    
