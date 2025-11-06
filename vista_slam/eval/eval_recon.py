import numpy as np
import os
import torch
import munch
from ..utils.slam_utils import compute_local_pointclouds

def load_data(output_foler, load_view_graph=True, 
              load_gt_depths=True, load_gt_poses=True, load_gt_intrinsic=True, 
              load_unscaled_depths=True,load_scales=True,
              load_intrinsics=True, load_confs=True, load_poses=True):
    data={}
    if load_view_graph:
        view_graph_npz = np.load(os.path.join(output_foler, "view_graph.npz"), allow_pickle=True)
        data['view_graph'] = view_graph_npz['view_graph'].item()  # {view: [connected_view, ...]}
        data['loop_min_dist'] = view_graph_npz['loop_min_dist'].item()
        data['view_names'] = view_graph_npz['view_names'].tolist()
    if load_gt_depths:
        data['gt_depths'] = np.load(os.path.join(output_foler, "gt_depths.npy"))
    if load_gt_poses:
        data['gt_poses'] = np.load(os.path.join(output_foler, "gt_poses.npy"))
    if load_gt_intrinsic:
        data['gt_intrinsic'] = np.load(os.path.join(output_foler, "gt_intrinsics.npy"))
    if load_unscaled_depths:
        data['unscaled_depths'] = np.load(os.path.join(output_foler, "depths.npy"))
    if load_scales:
        data['scales'] = np.load(os.path.join(output_foler, "scales.npy"))[...,None]  # [N, 1, 1]
    if load_intrinsics:
        data['intrinsics'] = np.load(os.path.join(output_foler, "intrinsics.npy"))
    if load_confs:
        confs_npz = np.load(os.path.join(output_foler, "confs.npz"))
        data['confs'] = confs_npz['confs']  # [N, H, W]
        data['conf_thres'] = confs_npz['thres'].item()  # float
    if load_poses:
        data['poses'] = np.load(os.path.join(output_foler, "trajectory.npy"))
    return munch.Munch(data)
    
def estimate_focus_length_with_error(pts3d, confidence):
    H, W, _ = pts3d.shape
    cx = W/2.0
    cy = H/2.0
    j, i = np.meshgrid(np.arange(W), np.arange(H))  # (x, y)
    u = j.astype(np.float32) - cx
    v = i.astype(np.float32) - cy

    X = pts3d[..., 0].reshape(-1)
    Y = pts3d[..., 1].reshape(-1)
    Z = pts3d[..., 2].reshape(-1)
    u = u.reshape(-1)
    v = v.reshape(-1)
    conf = confidence.reshape(-1)
    weights = np.clip(conf, a_min=1e-6, a_max=None)

    xz = np.nan_to_num(X / Z, nan=0.0, posinf=0.0, neginf=0.0)
    yz = np.nan_to_num(Y / Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Weighted least squares estimates
    fx_num = np.sum(weights * xz * u)
    fx_denom = np.sum(weights * xz**2)
    fx = fx_num / fx_denom

    fy_num = np.sum(weights * yz * v)
    fy_denom = np.sum(weights * yz**2)
    fy = fy_num / fy_denom

    # Residuals
    fx_residuals = u - fx * xz
    fy_residuals = v - fy * yz

    # Variance estimate of residuals
    fx_var = np.sum(weights * fx_residuals**2) / (np.sum(weights * xz**2)**2)
    fy_var = np.sum(weights * fy_residuals**2) / (np.sum(weights * yz**2)**2)

    fx_std = np.sqrt(fx_var)
    fy_std = np.sqrt(fy_var)

    return fx, fy, fx_std, fy_std

def transform_to_world_coordinates(cam_coords,poses):
    # Homogeneous coordinates
    cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[:,:,:,0:1])], dim=-1)  # (N, H, W, 4)

    # Transform to world coordinates
    points_world = torch.matmul(poses[:, None, None, :, :], cam_coords_h[..., None])  # (N, H, W, 4, 1)
    points_world = points_world.squeeze(-1)[..., :3]  # (N, H, W, 3)

    return points_world


def chamfer_distance_RMSE(pcd_ref, pcd_est, max_error):
    from pykdtree.kdtree import KDTree as pyKDTree
    ref, est = np.asarray(pcd_ref.points), np.asarray(pcd_est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    rmse_dist1 = np.sqrt(np.mean(dist1**2))
    rmse_dist2 = np.sqrt(np.mean(dist2**2))
    chamfer_dist = 0.5 * rmse_dist1 + 0.5 * rmse_dist2

    # print("dist1", rmse_dist1, np.mean(dist1), np.median(dist1), np.max(dist1))
    # print("dist2", rmse_dist2, np.mean(dist2), np.median(dist2), np.max(dist2))

    return chamfer_dist, rmse_dist1, rmse_dist2, dist1, dist2


def eval_recon(gt_depths, gt_poses, gt_intri,
               est_depths, est_poses, est_intris, est_masks, 
               rel_R, rel_t, rel_s):
    """
    gt_depths: N,H,W
    gt_poses: N,4,4
    gt_intri: 3,3
    est_local_pcls: N,H,W,3
    est_masks: N,H,W
    est_poses: N,4,4
    rel_R: 3,3
    rel_t: 3,
    rel_s: 1
    """
    s = rel_s
    r_a = rel_R
    t_a = rel_t.reshape(3, 1)

    import open3d as o3d
    assert len(gt_depths) == len(est_depths) == len(est_masks) == len(est_poses) == len(gt_poses)
    view_num = len(gt_depths)

    N, H, W = gt_depths.shape
    gt_local_pcls = compute_local_pointclouds(gt_depths, gt_intri)
    est_local_pcls = compute_local_pointclouds(est_depths,est_intris)

    gt_pcls = transform_to_world_coordinates(gt_local_pcls, gt_poses).cpu().numpy()
    est_pcls = transform_to_world_coordinates(est_local_pcls, est_poses).cpu().numpy()

    gt_masks = (gt_depths > 0).cpu().numpy().astype(bool)
    est_masks = est_masks.cpu().numpy().astype(bool)

    gt_pcls = gt_pcls[gt_masks]
    est_pcls = est_pcls[est_masks & gt_masks]

    pcd_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(est_pcls))
    pcd_est.paint_uniform_color([0.0, 0.0, 1.0])

    center = np.asarray(pcd_est.get_center(), dtype=np.float64)
    # scale point cloud and given initial estimate of transformation.
    points = np.asarray(pcd_est.points)
    scaled_points = ((s*r_a) @ points.T + t_a).T
    pcd_est.points = o3d.utility.Vector3dVector(scaled_points)

    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pcls))
    gt_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # Run ICP alignment.
    # Downsample both point clouds for ICP only.
    voxel_size = 0.05  # Adjust this depending on your scale and desired speed/accuracy tradeoff
    pcd_est_down = pcd_est.voxel_down_sample(voxel_size)
    gt_pcd_down = gt_pcd.voxel_down_sample(voxel_size)

    # Run ICP on downsampled point clouds
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_est_down,
        gt_pcd_down,
        max_correspondence_distance=0.1,  # Adjust to match voxel size
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply the transformation to the full-resolution source point cloud
    transformation = reg_p2p.transformation
    pcd_est.transform(transformation)

    chamfer_dist, rmse_acc, rmse_comp, dists1, dists2 = (
        chamfer_distance_RMSE(gt_pcd, pcd_est, max_error=0.5)
    )

    return rmse_acc, rmse_comp, chamfer_dist, gt_pcd.points, pcd_est.points


def eval_recon_from_saved_data(output_folder, rel_est_gt:list|None=None):
    '''
    rel_gt_est: None or [R, t, s] for the relative pose between the ground truth and estimated point clouds.
    '''
    
    data = load_data(f"{output_folder}",load_view_graph=False)
    gt_depths = torch.from_numpy(data.gt_depths).float()
    gt_poses = torch.from_numpy(data.gt_poses).float()
    gt_intri = torch.from_numpy(data.gt_intrinsic)
    
    conf_thres = data.conf_thres
    est_depths = torch.from_numpy(data.unscaled_depths * data.scales).float()  # [N, H, W]
    est_intris = torch.from_numpy(data.intrinsics).float()  # [N, 3, 3]
    est_masks = torch.from_numpy(data.confs > conf_thres).float()  # [N, H, W]
    est_poses = torch.from_numpy(data.poses).float()  # [N, 4, 4]
    
    if rel_est_gt is not None:
        rel_R, rel_t, rel_s = rel_est_gt
    else:
        from .eval_traj import align_traj
        rel_R, rel_t, rel_s, _, _ = align_traj(est_poses, gt_poses)
    
    rmse_acc, rmse_comp, chamfer_dist, gt_pcd, pcd_est = eval_recon(gt_depths, gt_poses, gt_intri,
                                                   est_depths, est_poses, est_intris, est_masks, 
                                                   rel_R, rel_t, rel_s)
    return rmse_acc, rmse_comp, chamfer_dist, gt_pcd, pcd_est



if __name__ == "__main__":
    output_folder = "output/test"
    dataset_name = "7scenes"
    # scene_names = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    scene_names = ["office"]
    
    acc = []
    comp = []
    chamfer = []
    for scene_name in scene_names:
        rmse_acc, rmse_comp, chamfer_dist, gt_pcd, pcd_est = \
            eval_recon_from_saved_data(f"{output_folder}/{dataset_name}_{scene_name}")
        acc.append(rmse_acc)
        comp.append(rmse_comp)
        chamfer.append(chamfer_dist)
        print(f"Results for {scene_name}: RMSE Accuracy: {rmse_acc:.4f}, RMSE Completeness: {rmse_comp:.4f}, Chamfer Distance: {chamfer_dist:.4f}")

    for i in range(len(scene_names)):
        print(scene_names[i])
        print("acc:",acc[i])
        print("comp:",comp[i])
        print("chamfer:",chamfer[i])
    
    print()
    print("avg:")       
    print("acc:",sum(acc)/len(acc))
    print("comp:",sum(comp)/len(comp))
    print("chamfer:",sum(chamfer)/len(chamfer))
    