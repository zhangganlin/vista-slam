import os.path as osp
import os
import cv2
import numpy as np
import math
import glob
import tqdm
import json

from .base.base_view_graph_dataset import BaseViewGraphDataset, is_good_type, transpose_to_landscape
from ..utils.image import imread_cv2   
from ..utils.geometry import depthmap_to_camera_coordinates


class ScanNetpp(BaseViewGraphDataset):
    def __init__(self,  
                 sensor_data_root='/data/yuzheng/data/scannetpp/train_val_scannetpp', 
                 view_graph_root='',
                 scene_name=None,
                 sample_min_interval=20, # any two sampled main_view should have a interval of at least sample_min_interval
                 neighbor_range=20,  # sample from [i-neighbor_range, i+neighbor_range]
                 neighbor_num=5,     # sample neighbor_num from [i-neighbor_range, i] and neighbor_num [i, i+neighbor_range]
                 loop_num=5,         # sample loop_num from the loop candidates
                 num_sample_per_scene=100,
                 non_loop=True,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_data_root = sensor_data_root
        self.view_graph_root = view_graph_root
        self.sample_min_interval = sample_min_interval        
        
        self.neighbor_range = neighbor_range
        self.neighbor_num = neighbor_num
        self.loop_num = loop_num
        self.non_loop = non_loop

        sub_list = sorted(os.listdir(view_graph_root))
        self.scene_names = []
        for sub in sub_list:
            sub_scene_list = os.listdir(osp.join(view_graph_root, sub))
            sub_scene_list = sorted(glob.glob(osp.join(view_graph_root, sub, '*imglist.txt')))
            sub_scene_list = [sub + '/' + osp.basename(x).rsplit("_imglist.txt", 1)[0] for x in sub_scene_list]
            self.scene_names.extend(sub_scene_list)

        self.num_sample_per_scene = num_sample_per_scene
        
        if self.split == 'train':
            self.scene_names = self.scene_names[:-30]
        elif self.split=='test':
            self.scene_names = self.scene_names[-30:]
        if scene_name is not None:
            assert self.split is None
            if isinstance(scene_name, list): 
                self.scene_names = scene_name
            else:
                assert isinstance(scene_name, str)
                self.scene_names = [scene_name]
        print(self)

    def txt_to_dict(self,viewgraph_path_base, scene_name):
        view_graph_path = osp.join(viewgraph_path_base, f'{scene_name}_viewgraph.txt')
        img_list_path = osp.join(viewgraph_path_base, f'{scene_name}_imglist.txt')
        with open(img_list_path, 'r') as f:
            img_list = [line.strip() for line in f]

        data = {}
        with open(view_graph_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(':', 1)
                main_img = img_list[int(key)]
                value_list = []
                for item in value.split(';'):
                    if item:
                        idx, uncertain = item.strip('()').split(',')
                        neighbor = img_list[int(idx)]
                        value_list.append(neighbor)
                data[main_img] = value_list
        return data

    def __len__(self):
        return len(self.scene_names)*self.num_sample_per_scene
    
    def _read_view(self, data_dir, view_name, intri, resolution, rng, frame_meta_data, filepath_index_mapping):
        try:
            frame_metadata = frame_meta_data[filepath_index_mapping.get(view_name+".JPG")]
            camera_pose = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
            camera_pose[:, 1:3] *= -1.0
            camera_pose[2, :] *= -1.0
            camera_pose[1, :] *= -1.0


            if not np.isfinite(camera_pose).all():
                # print(data_dir,view_name, "pose invalid")
                return False, None
            # Load RGB image
            rgb_image = imread_cv2(osp.join(data_dir, f"undistorted_images/{view_name}.JPG"))
            if not np.isfinite(rgb_image).all():
                # print(data_dir,view_name, "img invalid")
                return False, None
            
            assert np.isfinite(intri).all()
            # Load depthmap
            depthmap = imread_cv2(osp.join(data_dir, f"undistorted_depths/{view_name}.png"), cv2.IMREAD_UNCHANGED)
        except:
            return False, None
        depthmap = depthmap.astype(np.float32) / 1000
        depthmap[~np.isfinite(depthmap)] = 0  # invalid
        valid_mask = depthmap > 0
        if valid_mask.sum() == 0:
            # print(data_dir,view_name, "depth invalid")
            return False, None

        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, intri, resolution,
            w_edge= 0, h_edge=0,
            rng=rng, info=f"{data_dir}/{view_name}")

        view = {}

        width, height = rgb_image.size
        view['true_shape'] = np.int32((height, width))
        view['img'] = self.transform(rgb_image)    
        view['valid_mask'] = valid_mask
        view['camera_pose'] = camera_pose
        view['camera_intrinsics'] = intrinsics
        view['depthmap'] = depthmap

        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(view['depthmap'], view['camera_intrinsics'])

        view['pts3d_cam'] = pts3d_cam  #(H, W, 3)
        view['valid_mask'] = valid_mask & np.isfinite(pts3d_cam).all(axis=-1)
        view['label'] = f'{data_dir}/{view_name}'
        
        # check all datatypes
        for key, val in view.items():
            res, err_msg = is_good_type(key, val)
            assert res, f"{err_msg} with {key}={val} for view {data_dir}/{view_name}"

        transpose_to_landscape(view)
        # this allows to check whether the RNG is is the same state each time
        view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')       

        return True, view




    def sample_frames(self, data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                      frame_meta_data, filepath_index_mapping, attemp=0):
        # img_list: list of abs image names
        # data_dir: the directory of the imagscene_namees
        # scene_loop_candiates: dict, key: center_frame, value: list of loop candidates (image names)
        # return: center_frame, neighbors (list of abs path), curr_loop (list of abs path)    
        if attemp > 10:
            return None, None, None


        interval = self.sample_min_interval
        img_indices = list(range(len(img_list)))

        center_view_idx_candidates = img_indices[self.neighbor_range:-self.neighbor_range:interval]

        success = False
        center_attemp = 0
        while not success:
            center_index = rng.choice(center_view_idx_candidates)
            center_frame = osp.basename(img_list[center_index])
            center_frame_basename = center_frame.split(".")[0] 
            if scene_loop_candiates is None or self.non_loop:
                success, center_view = self._read_view(data_dir, center_frame_basename, intri, resolution, rng,
                                                       frame_meta_data, filepath_index_mapping)
                loop_candidates = []
            elif center_frame in scene_loop_candiates.keys():
                success, center_view = self._read_view(data_dir, center_frame_basename, intri, resolution, rng,
                                                       frame_meta_data, filepath_index_mapping)
                loop_candidates = scene_loop_candiates[center_frame]
            center_attemp += 1
            if center_attemp > 10: return None, None, None

        loop_views=[]
        loop_view_idx=[]

        failure_count = 0

        if len(loop_candidates) < self.loop_num:
            for l_c in loop_candidates:
                valid, l_view = self._read_view(data_dir, l_c.split(".")[0] , intri, resolution, rng,
                                                frame_meta_data, filepath_index_mapping)
                if valid: loop_views.append(l_view)
                else: failure_count += 1 
                if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                                                                 frame_meta_data, filepath_index_mapping, attemp+1)    
        else:
            rand_idx = rng.choice(len(loop_candidates), self.loop_num, replace=False)
            for l_idx in rand_idx:
                valid, l_view = self._read_view(data_dir, loop_candidates[l_idx].split(".")[0] , intri, resolution, rng,
                                                frame_meta_data, filepath_index_mapping)
                if valid: loop_views.append(l_view)
                else: failure_count += 1
                if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                                                                 frame_meta_data, filepath_index_mapping, attemp+1) 

        neigbor_candidates_idx_left = list(range(center_index-self.neighbor_range, center_index))
        neigbor_candidates_idx_right = list(range(center_index+1, center_index+self.neighbor_range))

        
        left_view_idx=[]
        right_view_idx=[]

        left_view=[]
        right_view=[]

        while len(left_view) < self.neighbor_num:
            left_idx = rng.choice(neigbor_candidates_idx_left)
            if left_idx in left_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[left_idx]).split(".")[0], intri, resolution, rng,
                                            frame_meta_data, filepath_index_mapping)
            if valid: 
                left_view.append(l_view)
                left_view_idx.append(left_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                                                             frame_meta_data, filepath_index_mapping, attemp+1) 
        while len(right_view) < self.neighbor_num:
            right_idx = rng.choice(neigbor_candidates_idx_right)
            if right_idx in right_view_idx: continue
            valid, r_view = self._read_view(data_dir, osp.basename(img_list[right_idx]).split(".")[0], intri, resolution, rng,
                                            frame_meta_data, filepath_index_mapping)
            if valid: 
                right_view.append(r_view)
                right_view_idx.append(right_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                                                             frame_meta_data, filepath_index_mapping, attemp+1) 
        while len(loop_views) < self.loop_num:
            loop_idx = rng.choice(neigbor_candidates_idx_left+neigbor_candidates_idx_right)
            if loop_idx in left_view_idx or loop_idx in right_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[loop_idx]).split(".")[0], intri, resolution, rng,
                                            frame_meta_data, filepath_index_mapping)
            if valid: 
                loop_views.append(l_view)
                loop_view_idx.append(loop_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, intri, rng,
                                                             frame_meta_data, filepath_index_mapping, attemp+1) 
        neighbors = left_view + right_view
        return center_view, neighbors, loop_views

    
    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_names[idx // self.num_sample_per_scene]

        cams_metadata_path = osp.join(self.sensor_data_root, scene_id, 'dslr/nerfstudio/transforms_undistorted.json')
        cams_meta_data = json.load(open(cams_metadata_path, "r"))
        fx, fy, cx, cy = cams_meta_data['fl_x'], cams_meta_data['fl_y'], cams_meta_data['cx'], cams_meta_data['cy']
        intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        frame_meta_data = cams_meta_data['frames']
        filepath_index_mapping = {frame["file_path"]: index for index, frame in enumerate(frame_meta_data)}



        data_path = osp.join(self.sensor_data_root, scene_id, 'dslr')
        with open(f"{self.view_graph_root}/{scene_id}_imglist.txt", "r") as f:
            img_list = f.read().splitlines()  # Removes newline characters
        scene_loop_candiates = self.txt_to_dict(self.view_graph_root, scene_id)

        center_frame, neighbors, loop_views = self.sample_frames(data_path, img_list, scene_loop_candiates, 
                                                                 resolution, intri, rng,
                                                                 frame_meta_data, filepath_index_mapping)
        if center_frame is None:
            print(f"Failed to sample frames for {scene_id}",force=True)

        assert center_frame is not None and neighbors is not None and loop_views is not None, f"Failed to sample frames for {scene_id}"

        views = {'main_view': center_frame, 'neighbor_views': neighbors, 'loop_views': loop_views}
        return views  
    

