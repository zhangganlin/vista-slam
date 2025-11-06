

if __name__ == "__main__":
    import argparse, yaml, munch, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tumrgbd.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to the dataset folder, e.g. '/path/to/tumrgbd'")
    parser.add_argument("--output", type=str, help="Path to the output folder")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    if args.output is not None:
        cfg['output_dir'] = args.output
    cfg['dataset_folder'] = args.dataset_folder
    cfg = munch.Munch(cfg)
    
    import torch
    import numpy as np
    from vista_slam.slam import OnlineSLAM
    from vista_slam.eval.eval_traj import full_traj_eval
    from vista_slam.datasets.slam_tumrgbd import SLAM_TUMRGBD
    import torch.backends.cudnn as cudnn
    from vista_slam.utils.slam_utils import FontColor, print_msg
    import tqdm
    from colorama import Fore
    print = print_msg
    
    # fix the seed
    seed = cfg.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True 
    
    dataset_name = "tumrgbd"
    scene_names = ['360','desk', 'desk2', 'floor', 'plant', 'room', 'rpy', 'teddy', 'xyz']
     
    output_folder = cfg['output_dir']
    os.makedirs(output_folder, exist_ok=True)
    
    traj_error =[]
    acc = []
    comp = []
    chamfer = []

    for scene_name in scene_names:
            
        dataset = SLAM_TUMRGBD(f"{cfg.dataset_folder}/rgbd_dataset_freiburg1_{scene_name}", resolution=(224,224))
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
        last = len(dataset)
        inputs = list(range(1,last,stride))

        first = True
        t_num = 0
        gt_poses = []
        if len(inputs) > cfg.max_view_num:
            inputs = list(torch.linspace(0, last - 1, steps=cfg.max_view_num).long())

        last_pcl = None
        last_conf = None

        gt_depth = []
        img_names = []

        pbar = tqdm.tqdm(total=last, desc=Fore.GREEN+f"[Progress] "+Fore.RESET+f"{dataset_name} {scene_name}", bar_format="{l_bar}{bar}|  [{n_fmt}/{total_fmt} frames]", position=0)
        for idx in range(len(inputs)):
            t = inputs[idx]
            pbar.n = int(t+1)
            pbar.refresh()
            data = dataset[t]
            img_names.append(data.img_name)
            img_gray = (data.gray.squeeze(0).numpy()*255).astype(np.uint8)
                        
            img_shape = torch.tensor(data.rgb.shape[1:3]).unsqueeze(0)
            img = data.rgb.unsqueeze(0).to(slam.device)
            input_value = {'rgb':img, 'shape':img_shape, 'gray':img_gray, 'view_name':data.img_name}
                        
            gt_depth.append(data.depth)
            K = data.intrinsic
            gt_pose = data.camera_pose
            gt_poses.append(gt_pose)

            is_optimized = slam.step(input_value, force_pgo=idx==len(inputs)-1)
            
            if first:
                first = False
                continue

            t_num += 1
        pbar.close()                          
        torch.cuda.empty_cache()
        est_poses = []
        for i in range(slam.view_num):
            view = slam.get_view(i, return_pose=True, return_depth=False, return_intri=False)
            est_poses.append(view.pose.cpu().numpy())

        slam.save_data_all(f"{output_folder}/{dataset_name}_{scene_name}",
                           save_view_graph=True, traj_name_postfix=None,
                           save_poses=True, save_images=True, 
                           save_depths=True, save_intrinsics=True, 
                           save_confs=True, save_ply=True,
                           gt_poses=gt_poses, gt_depths=gt_depth, gt_intrinsics=K)

        print("Evaluating trajectory ...", color=FontColor.EVAL)
        _,_, r_a, t_a, s, ape_statistics = full_traj_eval(est_poses, gt_poses,f"{output_folder}/{dataset_name}_{scene_name}", "traj")
        traj_error.append(ape_statistics['rmse'])

        output_str = "#"*35 +f"\nEvaluation for {dataset_name} scene {scene_name}:\n"
        output_str += "#"*10+"traj evaluation"+"#"*10+"\n"
        output_str += f"relative scale: {s}\n"
        output_str += f"relative rotation:\n{r_a}\n"
        output_str += f"relative translation:{t_a}\n"
        output_str += f"statistics:\n{ape_statistics}\n"
        
        output_str += "#"*35 + "\n"
        out_path=f'{output_folder}/{dataset_name}_{scene_name}/evaluation_results.txt'
        with open(out_path, 'w+') as fp:
            fp.write(output_str)
        print(output_str, color=FontColor.EVAL)
        
    for i in range(len(scene_names)):
        print(scene_names[i],color=FontColor.EVAL)
        print(f"traj: {traj_error[i]}",color=FontColor.EVAL)

    print()
    print("avg:", color=FontColor.EVAL)
    print(f"traj: {sum(traj_error)/len(traj_error)}",color=FontColor.EVAL)

