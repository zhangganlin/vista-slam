import numpy as np
import os.path as osp
import os
import glob
import cv2
import munch
import torchvision.transforms as tvf
import itertools
import pandas as pd
from .base.base_view_graph_dataset import BaseViewGraphDataset
from ..utils.image import imread_cv2   
from ..utils.geometry import depthmap_to_camera_coordinates

class SLAM_TUMRGBD(BaseViewGraphDataset):
    def __init__(self, path_to_scene, resolution=(224,224)):
        super().__init__(resolution=resolution)
        self.resolution = resolution
        self.input_folder = f"{path_to_scene}"
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)
        intri = self.parse_list(osp.join(path_to_scene, 'intrinsics.txt'))
        self.intri = intri.astype(np.float32)

    def parse_list(self, filepath):
        """ read list data """
        df = pd.read_csv(filepath, sep='\s+', header=None, comment='#')
        return df.values

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list)

        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w

            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, i):
        value = munch.Munch()
        camera_pose = self.poses[i].astype(np.float32)
        rgb_image = imread_cv2(self.color_paths[i])
        depthmap = imread_cv2(self.depth_paths[i], cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 5000.0
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
