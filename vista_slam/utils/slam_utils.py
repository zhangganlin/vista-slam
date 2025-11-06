import torch
import torch.nn.functional as F
from contextlib import contextmanager
import builtins
from colorama import Fore, Style
from tqdm import tqdm

def estimate_intrinsic_from_pts3d(pts3d: torch.Tensor, confidence: torch.Tensor, shared_intrinsic: bool = False):
    """
    Estimate camera intrinsic matrix (or matrices) from 3D point maps and confidence.

    Args:
        pts3d: [B, H, W, 3] - 3D points in camera coordinates.
        confidence: [B, H, W] - confidence per pixel.
        shared_intrinsic: bool - if True, estimate one shared intrinsic matrix for the whole batch.

    Returns:
        K: [3, 3] if shared, else [B, 3, 3] - intrinsic matrix (matrices).
    """
    B, H, W, _ = pts3d.shape
    cx = W / 2.0
    cy = H / 2.0

    # Generate pixel grid [H, W]
    j = torch.arange(W, device=pts3d.device)
    i = torch.arange(H, device=pts3d.device)
    v, u = torch.meshgrid(i, j, indexing='ij')  # v: [H, W], u: [H, W]
    u = u.float() - cx
    v = v.float() - cy
    u = u.reshape(1, -1)  # [1, HW]
    v = v.reshape(1, -1)

    # Flatten
    X = pts3d[..., 0].reshape(B, -1)
    Y = pts3d[..., 1].reshape(B, -1)
    Z = pts3d[..., 2].reshape(B, -1)
    weights = torch.clamp(confidence.reshape(B, -1), min=1e-6)

    xz = torch.nan_to_num(X / Z, nan=0.0, posinf=0.0, neginf=0.0)
    yz = torch.nan_to_num(Y / Z, nan=0.0, posinf=0.0, neginf=0.0)

    if shared_intrinsic:
        # Combine all batches
        xz_all = xz.reshape(-1)
        yz_all = yz.reshape(-1)
        u_all = u.expand(B, -1).reshape(-1)
        v_all = v.expand(B, -1).reshape(-1)
        w_all = weights.reshape(-1)

        fx_num = torch.sum(w_all * xz_all * u_all)
        fx_denom = torch.sum(w_all * xz_all ** 2)
        fx = fx_num / fx_denom

        fy_num = torch.sum(w_all * yz_all * v_all)
        fy_denom = torch.sum(w_all * yz_all ** 2)
        fy = fy_num / fy_denom

        K = torch.tensor([[fx, 0, cx],
                          [0,  fy, cy],
                          [0,   0,  1]], dtype=pts3d.dtype, device=pts3d.device)
        return K  # [3, 3]

    else:
        # Per-batch intrinsic
        fx_num = torch.sum(weights * xz * u, dim=1)
        fx_denom = torch.sum(weights * xz ** 2, dim=1)
        fx = fx_num / fx_denom

        fy_num = torch.sum(weights * yz * v, dim=1)
        fy_denom = torch.sum(weights * yz ** 2, dim=1)
        fy = fy_num / fy_denom

        K = torch.zeros(B, 3, 3, dtype=pts3d.dtype, device=pts3d.device)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K  # [B, 3, 3]


def compute_local_pointclouds(depths: torch.Tensor, intrinsics: torch.Tensor):
    """
    Computes local point clouds from depth maps and intrinsics.

    Args:
        depths: [N, H, W] depth maps.
        intrinsics: [3, 3] or [N, 3, 3] camera intrinsic matrices.

    Returns:
        pointclouds: [N, H, W, 3] 3D points in camera space.
    """
    N, H, W = depths.shape
    device = depths.device

    # Create meshgrid once
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pix_coords = torch.stack((x, y, ones), dim=-1).float()  # (H, W, 3)
    pix_coords = pix_coords.reshape(-1, 3)  # (H*W, 3)

    if intrinsics.ndim == 2:
        # Shared intrinsics: [3, 3]
        K_inv = torch.inverse(intrinsics)  # [3, 3]
        cam_coords = torch.matmul(K_inv, pix_coords.T).T  # (H*W, 3)
        cam_coords = cam_coords.reshape(1, H, W, 3).repeat(N, 1, 1, 1)  # [N, H, W, 3]

    elif intrinsics.ndim == 3:
        # Batched intrinsics: [N, 3, 3]
        K_inv = torch.inverse(intrinsics)  # [N, 3, 3]
        pix_coords = pix_coords[None, :, :, None]  # [1, H*W, 3, 1]
        pix_coords = pix_coords.expand(N, -1, -1, -1)  # [N, H*W, 3, 1]
        cam_coords = torch.matmul(K_inv[:, None], pix_coords)  # [N, H*W, 3, 1]
        cam_coords = cam_coords.squeeze(-1).reshape(N, H, W, 3)  # [N, H, W, 3]

    else:
        raise ValueError(f"Unsupported intrinsics shape: {intrinsics.shape}")

    # Scale by depth
    cam_coords = cam_coords * depths[..., None]  # [N, H, W, 3]
    return cam_coords


def depth_from_pointcloud_dot_batched(pointclouds: torch.Tensor, intrinsics: torch.Tensor):
    """
    Compute depth from batched 3D pointclouds and intrinsics using dot product with ray directions.

    Args:
        pointclouds: [B, H, W, 3] - 3D points in camera space
        intrinsics: [3, 3] or [B, 3, 3] - camera intrinsics

    Returns:
        depths: [B, H, W] - computed depth maps
    """
    B, H, W, _ = pointclouds.shape
    device = pointclouds.device

    # Create pixel grid [H, W, 3]
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device),
                          indexing='ij')
    ones = torch.ones_like(x)
    pix_coords = torch.stack((x, y, ones), dim=-1).float()  # [H, W, 3]
    pix_coords_flat = pix_coords.view(-1, 3)  # [H*W, 3]

    if intrinsics.ndim == 2:
        # Shared intrinsics [3, 3]
        K_inv = torch.inverse(intrinsics)  # [3, 3]
        rays = torch.matmul(K_inv, pix_coords_flat.T).T  # [H*W, 3]
        rays = rays.view(1, H, W, 3).expand(B, -1, -1, -1)  # [B, H, W, 3]
    elif intrinsics.ndim == 3:
        # Batched intrinsics [B, 3, 3]
        K_inv = torch.inverse(intrinsics)  # [B, 3, 3]
        pix = pix_coords_flat.T.unsqueeze(0).expand(B, 3, H*W)  # [B, 3, H*W]
        rays = torch.bmm(K_inv, pix).permute(0, 2, 1).reshape(B, H, W, 3)  # [B, H, W, 3]
    else:
        raise ValueError(f"Unsupported intrinsics shape: {intrinsics.shape}")

    # Normalize ray directions
    rays_unit = rays / torch.norm(rays, dim=-1, keepdim=True)  # [B, H, W, 3]

    # Dot product between each point and its ray
    depths = torch.sum(pointclouds * rays_unit, dim=-1)  # [B, H, W]

    return depths


def estimate_scale_with_depth_and_confidence(Di, Dj, ci, cj):
    """
    Estimate scale s such that Dj ≈ s * Di, using dual confidence weights.
    
    Args:
        Di, Dj: [H, W] or [N] depth maps
        ci, cj: [H, W] or [N] confidence maps

    Returns:
        s: scalar scale estimate
    """
    Di = Di.reshape(-1)
    Dj = Dj.reshape(-1)
    ci = ci.reshape(-1)
    cj = cj.reshape(-1)

    w = ci * cj  # [N]
    w = w.clamp(min=1e-6)  # avoid zero division

    numerator = torch.sum(w * Di * Dj)
    denominator = torch.sum(w * Di * Di)
    s = numerator / denominator
    return s


def compute_geo_valid_mask_batched(depth1, depth2, K1, K2, T1, T2, error_thres_rel):
    """
    Compute dense pixel correspondence from image1 to image2 in a batch.

    Inputs:
        depth1: [B, H, W]
        depth2: [B, H, W] (unused, for potential future filtering)
        K1, K2: [B, 3, 3]
        T1, T2: [B, 4, 4]
    Returns:
        correspondence_map: [B, H, W, 2] – pixel (u,v) in image1 -> (u2,v2) in image2
    """
    B, H, W = depth1.shape
    device = depth1.device

    # Step 1: Create meshgrid [H, W, 2]
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([uu, vv], dim=-1).float()  # [H, W, 2]
    uv = uv[None].expand(B, H, W, 2)  # [B, H, W, 2]

    # Step 2: Backproject to camera frame
    z = depth1  # [B, H, W]
    fx = K1[:, 0, 0][:, None, None]
    fy = K1[:, 1, 1][:, None, None]
    cx = K1[:, 0, 2][:, None, None]
    cy = K1[:, 1, 2][:, None, None]

    x = (uv[..., 0] - cx) * z / fx
    y = (uv[..., 1] - cy) * z / fy
    pts_cam1 = torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]

    # Step 3: Transform to world
    ones = torch.ones_like(z)[..., None]
    pts_cam1_h = torch.cat([pts_cam1, ones], dim=-1)  # [B, H, W, 4]
    pts_cam1_h_flat = pts_cam1_h.view(B, -1, 4).transpose(1, 2)  # [B, 4, HW]
    pts_world = (T1 @ pts_cam1_h_flat).transpose(1, 2)[..., :3]  # [B, HW, 3]

    # Step 4: Transform to camera 2
    T2_inv = torch.inverse(T2)  # [B, 4, 4]
    pts_world_h = torch.cat([pts_world, torch.ones_like(pts_world[..., :1])], dim=-1)  # [B, HW, 4]
    pts_world_h = pts_world_h.transpose(1, 2)  # [B, 4, HW]
    pts_cam2 = (T2_inv @ pts_world_h).transpose(1, 2)[..., :3]  # [B, HW, 3]

    # Step 5: Project to image 2
    x2 = pts_cam2[..., 0]
    y2 = pts_cam2[..., 1]
    z2 = pts_cam2[..., 2]

    fx2 = K2[:, 0, 0][:, None]
    fy2 = K2[:, 1, 1][:, None]
    cx2 = K2[:, 0, 2][:, None]
    cy2 = K2[:, 1, 2][:, None]

    u2 = fx2 * x2 / z2 + cx2
    v2 = fy2 * y2 / z2 + cy2

    uv2 = torch.stack([v2, u2], dim=-1).int()  # [B, HW, 2]
    # uv2 = uv2.view(B, H, W, 2)
    
    valid_mask = (uv2[...,0]>=0) & (uv2[...,0]<H) & (uv2[...,1]>=0) & (uv2[...,1]<W)
    uv2[~valid_mask] = 0
    depth_from_1 = z2
    batch_idx = torch.arange(B, device=depth2.device).view(B, 1).expand(B, H*W)
    depth_from_2 = depth2[batch_idx,uv2[...,0],uv2[...,1]]
    
    error = (depth_from_1 - depth_from_2).abs()
    valid_error = error[valid_mask]
    error_thres = torch.quantile(valid_error, error_thres_rel)
    error_mask = error < error_thres
    mask = (error_mask & valid_mask).view(B,H,W)

    return mask


def compute_symmetric_geo_valid_mask(depths, intri, relative_pose):
    """
    Vectorized computation of symmetric geometric valid masks between two depth maps.

    Args:
        depths: [2, H, W] tensor
        intri: [3, 3]
        relative_pose: [4, 4], transform from cam1 to cam2

    Returns:
        mask: [2, H, W] – forward and backward valid pixel masks
    """
    device = depths.device
    H, W = depths.shape[1:]
    K = intri.to(device)
    K_inv = torch.inverse(K)

    # Generate meshgrid in homogeneous coordinates
    u, v = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0).float()  # [3, H, W]
    uv1 = uv1.view(3, -1)  # [3, HW]

    # Backproject both depth maps to 3D (camera frame)
    pts1_cam = (K_inv @ uv1) * depths[0].view(1, -1)  # [3, HW]
    pts2_cam = (K_inv @ uv1) * depths[1].view(1, -1)  # [3, HW]

    # Homogeneous coordinates
    ones = torch.ones(1, pts1_cam.shape[1], device=device)
    pts1_cam_h = torch.cat([pts1_cam, ones], dim=0)  # [4, HW]
    pts2_cam_h = torch.cat([pts2_cam, ones], dim=0)  # [4, HW]

    # Transform to other camera frame
    T_1to2 = relative_pose.to(device)
    T_2to1 = torch.inverse(T_1to2)

    pts1_to_2 = T_1to2 @ pts1_cam_h  # [4, HW]
    pts2_to_1 = T_2to1 @ pts2_cam_h  # [4, HW]

    # Perspective projection
    def project(pts):
        pts = pts[:3]
        uv_proj = K @ pts
        uv_proj = uv_proj[:2] / (uv_proj[2:] + 1e-8)
        return uv_proj, pts[2]  # uv, z

    uv1_to_2, z1_proj = project(pts1_to_2)
    uv2_to_1, z2_proj = project(pts2_to_1)

    # Round to nearest integer for sampling
    uv1_to_2_ = uv1_to_2.round().long()
    uv2_to_1_ = uv2_to_1.round().long()

    valid_1 = (uv1_to_2_[0] >= 0) & (uv1_to_2_[0] < W) & (uv1_to_2_[1] >= 0) & (uv1_to_2_[1] < H)
    valid_2 = (uv2_to_1_[0] >= 0) & (uv2_to_1_[0] < W) & (uv2_to_1_[1] >= 0) & (uv2_to_1_[1] < H)

    # Sample depth values from the target depths
    idx1 = uv1_to_2_[:, valid_1]
    sampled_d2 = depths[1, idx1[1], idx1[0]]
    err1 = (sampled_d2 - z1_proj[valid_1]).abs()

    idx2 = uv2_to_1_[:, valid_2]
    sampled_d1 = depths[0, idx2[1], idx2[0]]
    err2 = (sampled_d1 - z2_proj[valid_2]).abs()

    # Error thresholds
    thres1 = 2*err1.median() if err1.numel() > 0 else 1e10
    thres2 = 2*err2.median() if err2.numel() > 0 else 1e10

    mask1_flat = torch.zeros(H * W, dtype=torch.bool, device=device)
    mask2_flat = torch.zeros(H * W, dtype=torch.bool, device=device)

    mask1_flat[valid_1] = err1 < thres1
    mask2_flat[valid_2] = err2 < thres2

    return torch.stack([mask1_flat.view(H, W), mask2_flat.view(H, W)], dim=0)  # [2, H, W]


def view_consistency_check(depth, intrinsics, poses, threshold=0.05):
    """
    For each frame, unproject depth to world, reproject to ±2 neighbor frames,
    and count number of views where depth reprojection agrees (within threshold).
    
    Inputs:
        depth:      [n, H, W]         depth maps
        intrinsics: [n, 3, 3]         intrinsic matrices
        poses:      [n, 4, 4]         camera-to-world matrices
        threshold:  float             acceptable depth error (e.g., 0.05m)
    Returns:
        count_map:  [n, H, W]         number of views agreeing per pixel
    """
    n, H, W = depth.shape
    device = depth.device
    count_map = torch.zeros_like(depth, dtype=torch.int32)

    # Create mesh grid once
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixel_grid = torch.stack([x, y, torch.ones_like(x)], dim=0).float()  # [3, H, W]
    pixel_grid = pixel_grid[None].expand(n, -1, -1, -1)  # [n, 3, H, W]

    for i in range(n):
        d_i = depth[i]  # [H, W]
        K_i = intrinsics[i]  # [3, 3]
        T_i = poses[i]  # [4, 4]
        K_i_inv = torch.inverse(K_i)

        # Unproject to world
        pix = pixel_grid[i].reshape(3, -1)  # [3, H*W]
        cam = K_i_inv @ pix  # [3, H*W]
        cam = cam * d_i.reshape(1, -1)  # scale by depth
        cam_h = torch.cat([cam, torch.ones_like(cam[:1])], dim=0)  # [4, H*W]
        world = T_i @ cam_h  # [4, H*W]
        world = world[:3].T.unsqueeze(0)  # [1, H*W, 3]

        valid_count = torch.zeros(H*W, device=device)

        for j in range(max(0, i - 4), min(n, i + 5)):
            if j == i:
                continue
            K_j = intrinsics[j]
            T_j = poses[j]
            T_j_inv = torch.inverse(T_j)

            # Project world points to j
            world_h = torch.cat([world, torch.ones_like(world[..., :1])], dim=-1)  # [1, H*W, 4]
            cam_j = torch.bmm(world_h, T_j_inv.T.unsqueeze(0))  # [1, H*W, 4]
            cam_j = cam_j[..., :3]

            z = cam_j[..., 2].clamp(min=1e-6)
            uv = torch.bmm(cam_j, K_j.T.unsqueeze(0))  # [1, H*W, 3]
            uv = (uv[..., :2] / uv[..., 2:]).reshape(H, W, 2)  # [H, W, 2]

            # Normalize grid for sampling
            uv_norm = uv.clone()
            uv_norm[..., 0] = (uv_norm[..., 0] / (W - 1)) * 2 - 1
            uv_norm[..., 1] = (uv_norm[..., 1] / (H - 1)) * 2 - 1
            uv_norm = uv_norm.unsqueeze(0)  # [1, H, W, 2]

            # Sample target depth
            depth_j = depth[j].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            sampled_d = F.grid_sample(depth_j, uv_norm, mode='bilinear', align_corners=True)
            sampled_d = sampled_d.view(-1)  # [H*W]

            # Check depth difference
            depth_proj = z.view(-1)
            error = torch.abs(sampled_d - depth_proj)
            agree = (error < threshold) & (depth_proj > 0)
            valid_count += agree.int()

        count_map[i] = valid_count.view(H, W)

    return count_map

        
class FontColor(object):
    PoseGraphOpt=Fore.CYAN
    LoopClosure=Fore.BLUE
    EdgeReject=Fore.YELLOW
    INFO=Fore.GREEN
    WARNING=Fore.RED
    EVAL=Fore.MAGENTA
    RESET=Style.RESET_ALL

def get_msg_prefix(color):
    if color == FontColor.PoseGraphOpt:
        msg_prefix = color + "[PoseGraphOpt] " + Style.RESET_ALL
    elif color == FontColor.LoopClosure:
        msg_prefix = color + "[LoopClosure] " + Style.RESET_ALL
    elif color == FontColor.EdgeReject:
        msg_prefix = color + "[EdgeReject] " + Style.RESET_ALL
    elif color == FontColor.INFO:
        msg_prefix = color + "[INFO] " + Style.RESET_ALL
    elif color == FontColor.WARNING:
        msg_prefix = color + "[WARNING] " + Style.RESET_ALL
    elif color == FontColor.EVAL:
        msg_prefix = color + "[EVAL] " + Style.RESET_ALL
    else:
        msg_prefix = Style.RESET_ALL
    return msg_prefix

def print_msg(msg='', color=Style.RESET_ALL, end="\n"):
    msg_prefix = get_msg_prefix(color)
    tqdm.write(msg_prefix + msg, end=end)
    
@contextmanager
def suppress_specific_print(substr: str, color=Style.RESET_ALL):
    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if substr in msg:
            return  # suppress
        return print_msg(msg, color=color)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print