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

class SLAM_Scannet(BaseViewGraphDataset):
    def __init__(self, path_to_scene, resolution=(224,224)):
        super().__init__(resolution=resolution)
        self.resolution = resolution
        self.input_folder = f"{path_to_scene}/sensor_data"
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.color.jpg')))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.depth.png')))
        self.pose_paths = sorted(glob.glob(os.path.join(
            self.input_folder, '*.pose.txt')))
        self.n_img = len(self.color_paths)
        intri_path = osp.join(path_to_scene, 'intrinsic/intrinsic_depth.txt')
        self.intri = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

    def __getitem__(self, i):
        value = munch.Munch()
        camera_pose = np.loadtxt(self.pose_paths[i]).astype(np.float32)
        rgb_image = imread_cv2(self.color_paths[i])
        depthmap = imread_cv2(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000
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

        return value

    def __len__(self):
        return self.n_img
