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


class ARKitScene(BaseViewGraphDataset):
    def __init__(self,  
                 sensor_data_root='/data/yuzheng/data/scannetpp/train_val_scannetpp', 
                 view_graph_root='',
                 scene_name=None,
                 sample_min_interval=20, # any two sampled main_view should have a interval of at least sample_min_interval
                 neighbor_range=50,  # sample from [i-neighbor_range, i+neighbor_range]
                 neighbor_num=5,     # sample neighbor_num from [i-neighbor_range, i] and neighbor_num [i, i+neighbor_range]
                 loop_num=5,         # sample loop_num from the loop candidates
                 num_sample_per_scene=100,
                 non_loop=False,
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

        self.scene_names.remove('train_3/41159584')   # this scene is too short to sample
        
        self.num_sample_per_scene = num_sample_per_scene
        
        self.traj_cache = {}
        
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
    
    def _read_view(self, data_dir, view_name, resolution, rng, poses_from_traj):
        try:
            frame_id = view_name.split("_")[1]
            scene_id = view_name.split("_")[0]
            camera_pose = self.get_pose(frame_id, poses_from_traj).astype(np.float32)
            if not np.isfinite(camera_pose).all():
                # print(data_dir,view_name, "pose invalid")
                return False, None
            # Load RGB image
            rgb_image = imread_cv2(osp.join(data_dir, f"lowres_wide/{view_name}.png"))
            if not np.isfinite(rgb_image).all():
                # print(data_dir,view_name, "img invalid")
                return False, None
            
            intri = self.get_intrinsic(osp.join(data_dir, "lowres_wide_intrinsics"), scene_id, frame_id).astype(np.float32)
            assert np.isfinite(intri).all()
            # Load depthmap
            depthmap = imread_cv2(osp.join(data_dir, f"lowres_depth/{view_name}.png"), cv2.IMREAD_UNCHANGED)
            
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




    def sample_frames(self, data_dir, img_list, scene_loop_candiates, resolution, rng,
                      poses_from_traj, attemp=0):
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
            center_frame_basename = center_frame.split(".png")[0] 
            if scene_loop_candiates is None or self.non_loop:
                success, center_view = self._read_view(data_dir, center_frame_basename, resolution, rng,
                                                       poses_from_traj)
                loop_candidates = []
            elif center_frame in scene_loop_candiates.keys():
                success, center_view = self._read_view(data_dir, center_frame_basename, resolution, rng,
                                                       poses_from_traj)
                loop_candidates = scene_loop_candiates[center_frame]
            center_attemp += 1
            if center_attemp > 10: return None, None, None

        loop_views=[]
        loop_view_idx=[]

        failure_count = 0

        if len(loop_candidates) < self.loop_num:
            for l_c in loop_candidates:
                valid, l_view = self._read_view(data_dir, l_c.split(".png")[0], resolution, rng,
                                                poses_from_traj)
                if valid: loop_views.append(l_view)
                else: failure_count += 1 
                if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, rng,
                                                                 poses_from_traj, attemp+1)    
        else:
            rand_idx = rng.choice(len(loop_candidates), self.loop_num, replace=False)
            for l_idx in rand_idx:
                valid, l_view = self._read_view(data_dir, loop_candidates[l_idx].split(".png")[0], resolution, rng,
                                                poses_from_traj)
                if valid: loop_views.append(l_view)
                else: failure_count += 1
                if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, rng,
                                                                 poses_from_traj, attemp+1) 

        neigbor_candidates_idx_left = list(range(center_index-self.neighbor_range, center_index))
        neigbor_candidates_idx_right = list(range(center_index+1, center_index+self.neighbor_range))

        
        left_view_idx=[]
        right_view_idx=[]

        left_view=[]
        right_view=[]

        while len(left_view) < self.neighbor_num:
            left_idx = rng.choice(neigbor_candidates_idx_left)
            if left_idx in left_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[left_idx]).split(".png")[0], resolution, rng,
                                            poses_from_traj)
            if valid: 
                left_view.append(l_view)
                left_view_idx.append(left_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, rng,
                                                             poses_from_traj, attemp+1) 
        while len(right_view) < self.neighbor_num:
            right_idx = rng.choice(neigbor_candidates_idx_right)
            if right_idx in right_view_idx: continue
            valid, r_view = self._read_view(data_dir, osp.basename(img_list[right_idx]).split(".png")[0], resolution, rng,
                                            poses_from_traj)
            if valid: 
                right_view.append(r_view)
                right_view_idx.append(right_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, rng,
                                                             poses_from_traj, attemp+1) 
        while len(loop_views) < self.loop_num:
            loop_idx = rng.choice(neigbor_candidates_idx_left+neigbor_candidates_idx_right)
            if loop_idx in left_view_idx or loop_idx in right_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[loop_idx]).split(".png")[0], resolution, rng,
                                            poses_from_traj)
            if valid: 
                loop_views.append(l_view)
                loop_view_idx.append(loop_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_loop_candiates, resolution, rng,
                                                             poses_from_traj, attemp+1) 
        neighbors = left_view + right_view
        return center_view, neighbors, loop_views

    
    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_names[idx // self.num_sample_per_scene]
        data_path = osp.join(self.sensor_data_root, scene_id)
        with open(f"{self.view_graph_root}/{scene_id}_imglist.txt", "r") as f:
            img_list = f.read().splitlines()  # Removes newline characters
            
            
        pose_path = os.path.join(self.sensor_data_root, scene_id, 'lowres_wide.traj')
        poses_from_traj = {}
        
        if scene_id not in self.traj_cache.keys():
            with open(pose_path, "r", encoding="utf-8") as f:
                traj = f.readlines()
            for line in traj:
                poses_from_traj[f"{round(float(line.split(' ')[0]), 3):.3f}"] = np.array(
                    self.traj_string_to_matrix(line)[1].tolist()
                )
            self.traj_cache[scene_id] = poses_from_traj
        else:
            poses_from_traj = self.traj_cache[scene_id]


        scene_loop_candiates = self.txt_to_dict(self.view_graph_root, scene_id)
        center_frame, neighbors, loop_views = self.sample_frames(data_path, img_list, scene_loop_candiates, 
                                                                 resolution, rng,
                                                                 poses_from_traj)
        if center_frame is None:
            print(f"Failed to sample frames for {scene_id}",force=True)

        assert center_frame is not None and neighbors is not None and loop_views is not None, f"Failed to sample frames for {scene_id}"

        views = {'main_view': center_frame, 'neighbor_views': neighbors, 'loop_views': loop_views}
        return views  
    
    def traj_string_to_matrix(self, traj_string):
        """convert traj_string into translation and rotation matrices
        Args:
            traj_string: A space-delimited file where each line represents a camera position at a particular timestamp.
            The file has seven columns:
            * Column 1: timestamp
            * Columns 2-4: rotation (axis-angle representation in radians)
            * Columns 5-7: translation (usually in meters)
        Returns:
            ts: translation matrix
            Rt: rotation matrix
        """
        tokens = traj_string.split()
        assert len(tokens) == 7
        ts = tokens[0]
        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))  # type: ignore
        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)
        return (ts, Rt)

    def get_intrinsic(self, intrinsics_dir, video_id,frame_id):
        '''
        Nerfstudio
        '''
        intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{frame_id}.pincam")

        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam")

        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam")

        _, _, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)
        intrinsic = np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        return intrinsic

    def get_pose(self, frame_id, poses_from_traj):
        frame_pose = None
        if str(frame_id) in poses_from_traj:
            frame_pose = np.array(poses_from_traj[str(frame_id)])
        else:
            for my_key in poses_from_traj:
                if abs(float(frame_id) - float(my_key)) < 0.1:
                    frame_pose = np.array(poses_from_traj[str(my_key)])
        assert frame_pose is not None
        # frame_pose[0:3, 1:3] *= -1
        # frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
        # frame_pose[2, :] *= -1
        return frame_pose
