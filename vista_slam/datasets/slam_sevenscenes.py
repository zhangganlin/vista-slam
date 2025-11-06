import numpy as np
import os.path as osp
import os
import glob
import cv2
import munch
import torchvision.transforms as tvf

from .base.base_view_graph_dataset import BaseViewGraphDataset
from ..utils.image import imread_cv2   
from ..utils.geometry import depthmap_to_camera_coordinates



def rgb_pixels_to_depth_map(K_rgb, K_depth, T_rgb_to_depth, depth_img, H_rgb, W_rgb):
    """
    Project depth image into RGB frame, get depth value for each RGB pixel.

    Parameters:
        K_rgb:     (3, 3) intrinsic matrix of RGB camera
        K_depth:   (3, 3) intrinsic matrix of depth camera
        T_rgb_to_depth: (4, 4) transformation from RGB to depth camera
        depth_img: (H_d, W_d) depth image (in meters)
        H_rgb, W_rgb: height and width of RGB image

    Returns:
        depth_rgb: (H_rgb, W_rgb) depth image in RGB camera frame
    """
    # Prepare output
    depth_rgb = np.full((H_rgb, W_rgb), np.nan, dtype=np.float32)

    # Inverse intrinsics
    K_rgb_inv = np.linalg.inv(K_rgb)

    # Decompose transformation
    R = T_rgb_to_depth[:3, :3]
    t = T_rgb_to_depth[:3, 3:]

    H_d, W_d = depth_img.shape

    # Prepare meshgrid of RGB pixel coordinates
    u, v = np.meshgrid(np.arange(W_rgb), np.arange(H_rgb))  # shape (H, W)
    ones = np.ones_like(u)
    pix_rgb = np.stack([u, v, ones], axis=-1).reshape(-1, 3).T  # (3, N)

    # Get rays in RGB camera frame
    rays_rgb = K_rgb_inv @ pix_rgb  # shape (3, N)

    # Transform rays to depth frame
    rays_depth = R @ rays_rgb  # shape (3, N)

    # Project rays into depth image
    p_depth = rays_depth + t  # assume unit length (λ=1), we'll correct with λ later
    proj = K_depth @ p_depth
    u_d = proj[0, :] / proj[2, :]
    v_d = proj[1, :] / proj[2, :]

    # Round and mask valid projection
    u_d = np.round(u_d).astype(np.int32)
    v_d = np.round(v_d).astype(np.int32)
    valid = (
        (u_d >= 0) & (u_d < W_d) &
        (v_d >= 0) & (v_d < H_d) &
        (rays_depth[2, :] != 0)
    )

    # Fetch depth values
    d_img = np.zeros(rays_rgb.shape[1], dtype=np.float32)
    d_img[valid] = depth_img[v_d[valid], u_d[valid]]

    # Compute λ = (d_img - t_z) / ray_depth_z
    lambda_vals = np.zeros_like(d_img)
    lambda_vals[valid] = (d_img[valid] - t[2, 0]) / rays_depth[2, valid]

    # Get actual point in RGB frame: λ * ray_rgb
    points_rgb = rays_rgb * lambda_vals[None, :]

    # Get z-coordinate in RGB frame as depth
    depth_vals = points_rgb[2, :]

    # Fill output depth map
    depth_rgb = depth_vals.reshape(H_rgb, W_rgb)
    depth_rgb[~valid.reshape(H_rgb, W_rgb)] = np.nan

    return depth_rgb


class SLAM_SevenScenes(BaseViewGraphDataset):
    def __init__(self, path_to_scene, resolution=(224,224)):
        super().__init__(resolution=resolution)
        self.resolution = resolution
        self.input_folder = f"{path_to_scene}"
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.color.png')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.depth.png')))
        self.pose_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.pose.txt')))
        self.n_img = len(self.color_paths)
        fx, fy, cx, cy = 532.57, 531.54, 320, 240
        fx_d, fy_d = 598.84, 587.62
        self.intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.intri_depth = np.array([[fx_d, 0, cx], [0, fy_d, cy], [0, 0, 1]], dtype=np.float32)
        self.T_rgb_d = np.array([[1,0,0,0.023449],
                                 [0,1,0,0.006177],
                                 [0,0,1,0.010525],
                                 [0,0,0,1]],dtype = np.float32)

    def __getitem__(self, i):
        value = munch.Munch()
        camera_pose = np.loadtxt(self.pose_paths[i]).astype(np.float32)
        rgb_image = imread_cv2(self.color_paths[i])
        depthmap = imread_cv2(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32)
        depthmap[depthmap == 65535] = 0
        depthmap = depthmap/ 1000.0
        depthmap[~np.isfinite(depthmap)] = 0  # invalid
        depthmap[depthmap > 4.5] = 0 # depth range of kinect is 0.5 - 4.5m

        depthmap = rgb_pixels_to_depth_map(self.intri,self.intri_depth,self.T_rgb_d,depthmap,480,640)
        depthmap[~np.isfinite(depthmap)] = 0  # invalid
                
        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))
        rgb_image, depthmap, intrinsic = self._crop_resize_if_necessary(
            rgb_image, depthmap, self.intri, self.resolution,
            w_edge=10, h_edge=10)
        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, intrinsic)
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ImgGray = tvf.Compose([tvf.ToTensor(), tvf.Grayscale(num_output_channels=1)])
        
        value['gray'] = ImgGray(rgb_image)
        value['rgb'] = ImgNorm(rgb_image)
        value['depth'] = depthmap
        value['intrinsic'] = intrinsic
        value['camera_pose'] = camera_pose
        value['pts3d_cam'] = pts3d_cam
        value['img_name'] = osp.basename(self.color_paths[i])

        return value

    def __len__(self):
        return self.n_img
