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

class SLAM_image_only(BaseViewGraphDataset):
    def __init__(self, image_paths, resolution=(224,224)):
        super().__init__(resolution=resolution)
        self.resolution = resolution
        self.color_paths = sorted(image_paths)
        self.n_img = len(self.color_paths)
        self.ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.ImgGray = tvf.Compose([tvf.ToTensor(), tvf.Grayscale(num_output_channels=1)])

    def process_image(self, rgb_image, img_name):
        """ Crop and resize an image to the desired resolution.
        """
        value = munch.Munch()
        rgb_image = self._crop_resize_if_necessary_image_only(
            rgb_image, self.resolution, w_edge=10, h_edge=10)

        value['gray'] = self.ImgGray(rgb_image)
        value['rgb'] = self.ImgNorm(rgb_image)
        value['img_name'] = osp.basename(img_name)

        return value
        

    def __getitem__(self, i):
        rgb_image = imread_cv2(self.color_paths[i])
        img_name = osp.basename(self.color_paths[i])
        value = self.process_image(rgb_image, img_name)
        return value

    def __len__(self):
        return self.n_img
