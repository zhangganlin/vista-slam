import os.path as osp
import os
import cv2
import numpy as np
import math
import glob
import tqdm

from .base.base_view_graph_dataset import BaseViewGraphDataset, is_good_type, transpose_to_landscape
from ..utils.image import imread_cv2   
from ..utils.geometry import depthmap_to_camera_coordinates


class Replica(BaseViewGraphDataset):
    def __init__(self,  
                 sensor_data_root='/storage/user/zga/datasets/Replica_full/replica_rendering_a', 
                 scene_name=None,
                 sample_min_interval=20, # any two sampled main_view should have a interval of at least sample_min_interval
                 neighbor_range=20,  # sample from [i-neighbor_range, i+neighbor_range]
                 neighbor_num=5,     # sample neighbor_num from [i-neighbor_range, i] and neighbor_num [i, i+neighbor_range]
                 loop_num=5,         # sample loop_num from the loop candidates
                 num_sample_per_scene=50,
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_data_root = sensor_data_root
        self.sample_min_interval = sample_min_interval        
        
        self.neighbor_range = neighbor_range
        self.neighbor_num = neighbor_num
        self.loop_num = loop_num

        self.scene_names = [f for f in os.listdir(sensor_data_root) if osp.isdir(osp.join(sensor_data_root, f))]


        self.num_sample_per_scene = num_sample_per_scene
        
        if self.split == 'train':
            self.scene_names.remove('room_0')  # for testing
        elif self.split=='test':
            self.scene_names = ['room_0']  # for testing
        if scene_name is not None:
            assert self.split is None
            if isinstance(scene_name, list): 
                self.scene_names = scene_name
            else:
                assert isinstance(scene_name, str)
                self.scene_names = [scene_name]
        self.intrinsics = {}
        self.poses = {}
        self.img_list = {}
        self._load_camera()
        print(self)

    def _load_camera(self):
        for scene_name in self.scene_names:
            scene_path = osp.join(self.sensor_data_root, scene_name)
            camera_path = osp.join(scene_path, 'camera.txt')
            with open(camera_path, 'r') as f:
                lines = f.readlines()
            w_h_fx_fy = lines[1]
            width, height, fx, fy = map(float, w_h_fx_fy.strip().split(','))
            cx = (width-1) / 2.0
            cy = (height-1) / 2.0
            intrinsics = np.array([[fx,  0.0,   cx],
                                   [0.0,  fy,   cy],
                                   [0.0, 0.0,  1.0]])
            self.intrinsics[scene_name] = intrinsics.astype(np.float32)

            traj_path = osp.join(scene_path, 'camera_trajectory.txt')
            poses = []
            with open(traj_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                poses.append(c2w)  # swap X and Z axes
            poses = np.array(poses).astype(np.float32)
            self.poses[scene_name] = poses

            img_list = sorted(glob.glob(osp.join(scene_path, 'frame*.jpg')))
            self.img_list[scene_name] = img_list


    def __len__(self):
        return len(self.scene_names)*self.num_sample_per_scene
    
    def _read_view(self, data_dir, view_name, intri, pose, resolution, rng):
        try:
            camera_pose = pose
            if not np.isfinite(camera_pose).all() or camera_pose.shape != (4, 4):
                # print(data_dir,view_name, "pose invalid")
                return False, None
            # Load RGB image
            rgb_image = imread_cv2(osp.join(data_dir, f"{view_name}.jpg"))
            if not np.isfinite(rgb_image).all():
                # print(data_dir,view_name, "img invalid")
                return False, None
            
            assert np.isfinite(intri).all()
            # Load depthmap
            depthmap = imread_cv2(osp.join(data_dir, f"{view_name.replace('frame', 'depth')}.png"), cv2.IMREAD_UNCHANGED)
        except:
            return False, None
        depthmap = depthmap.astype(np.float32) / 6553.5
        depthmap[~np.isfinite(depthmap)] = 0  # invalid
        depthmap[depthmap > 50.0] = 0
        valid_mask = depthmap > 0


        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, intri, resolution,
            w_edge=0, h_edge=0,
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
        if view['valid_mask'].sum() == 0:
            # print(data_dir,view_name, "depth invalid")
            return False, None
        view['label'] = f'{data_dir}/{view_name}'
        
        # check all datatypes
        for key, val in view.items():
            res, err_msg = is_good_type(key, val)
            assert res, f"{err_msg} with {key}={val} for view {data_dir}/{view_name}"

        transpose_to_landscape(view)
        # this allows to check whether the RNG is is the same state each time
        view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')       

        return True, view




    def sample_frames(self, data_dir, img_list, scene_id, resolution, intri, rng, attemp=0):
        # img_list: list of abs image names
        # data_dir: the directory of the imagscene_namees
        # scene_loop_candiates: dict, key: center_frame, value: list of loop candidates (image names)
        # return: center_frame, neighbors (list of abs path), curr_loop (list of abs path)    
        if attemp > 10:
            return None, None, None


        interval = self.sample_min_interval
        img_indices = list(range(len(img_list)))
        poses = self.poses[scene_id]

        center_view_idx_candidates = img_indices[self.neighbor_range:-self.neighbor_range:interval]
        if len(center_view_idx_candidates) < 1:
            return None, None, None
        
        
        success = False
        center_attemp = 0
        while not success:
            center_index = rng.choice(center_view_idx_candidates)
            center_frame = osp.basename(img_list[center_index])
            center_frame_basename = center_frame.split(".")[0] 
            pose = poses[center_index]
            success, center_view = self._read_view(data_dir, center_frame_basename, intri, pose, resolution, rng)
            center_attemp += 1
            if center_attemp > 10: return None, None, None

        loop_views=[]
        loop_view_idx=[]

        failure_count = 0


        neigbor_candidates_idx_left = list(range(center_index-self.neighbor_range, center_index))
        neigbor_candidates_idx_right = list(range(center_index+1, center_index+self.neighbor_range))

        
        left_view_idx=[]
        right_view_idx=[]

        left_view=[]
        right_view=[]

        while len(left_view) < self.neighbor_num:
            left_idx = rng.choice(neigbor_candidates_idx_left)
            pose = poses[left_idx]
            if left_idx in left_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[left_idx]).split(".")[0], intri, pose, resolution, rng)
            if valid: 
                left_view.append(l_view)
                left_view_idx.append(left_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_id, resolution, intri, rng, attemp+1) 
        while len(right_view) < self.neighbor_num:
            right_idx = rng.choice(neigbor_candidates_idx_right)
            pose = poses[right_idx]
            if right_idx in right_view_idx: continue
            valid, r_view = self._read_view(data_dir, osp.basename(img_list[right_idx]).split(".")[0], intri, pose, resolution, rng)
            if valid: 
                right_view.append(r_view)
                right_view_idx.append(right_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_id, resolution, intri, rng, attemp+1) 
        while len(loop_views) < self.loop_num:
            loop_idx = rng.choice(neigbor_candidates_idx_left+neigbor_candidates_idx_right)
            pose = poses[loop_idx]
            if loop_idx in left_view_idx or loop_idx in right_view_idx: continue
            valid, l_view = self._read_view(data_dir, osp.basename(img_list[loop_idx]).split(".")[0], intri, pose, resolution, rng)
            if valid: 
                loop_views.append(l_view)
                loop_view_idx.append(loop_idx)
            else: failure_count += 1 
            if failure_count > 10: return self.sample_frames(data_dir, img_list, scene_id, resolution, intri, rng, attemp+1) 
        neighbors = left_view + right_view
        return center_view, neighbors, loop_views

    
    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_names[idx // self.num_sample_per_scene]

        intri = self.intrinsics[scene_id]
        data_path = osp.join(self.sensor_data_root, scene_id)
        img_list = self.img_list[scene_id]

        center_frame, neighbors, loop_views = self.sample_frames(data_path, img_list, scene_id,
                                                                 resolution, intri, rng)
        if center_frame is None:
            # print(f"Failed to sample frames for {scene_id}",force=True)
            print(f"Failed to sample frames for {scene_id}")
            new_idx = rng.choice(len(self.scene_names))
            return self._get_views(new_idx, resolution, rng)


        assert center_frame is not None and neighbors is not None and loop_views is not None, f"Failed to sample frames for {scene_id}"

        views = {'main_view': center_frame, 'neighbor_views': neighbors, 'loop_views': loop_views}
        return views  
    



# for testing

if __name__ == "__main__":
    import rerun as rr

    dataset = Replica(split="train", resolution=(224,224), 
                      scene_name=None)
    rr.init("replica_test",spawn=False)
    # rr.connect_tcp("131.159.19.231:9876")
    rr.connect_tcp("0.0.0.0:9876")


    rr.log(f"/world",rr.Transform3D())
    # for idx in np.random.permutation(len(dataset))[:10]:
    for idx in range(len(dataset))[:10]:
        views_dict = dataset[(idx,0)]
        rr.set_time_sequence("sample",idx)
        print(views_dict['main_view']['label'])
        views = [views_dict['main_view']] + views_dict['neighbor_views'] + views_dict['loop_views']

        for i, view in enumerate(views):
            img = view['img']
            img = (img+1.0)/2.0
            pts3d = np.array(view['pts3d_cam']).reshape(-1,3)
            
            pose = view['camera_pose']
                        
            rr.log(f"/world/cam_{i}",rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]))
            rr.log(f"/world/cam_{i}/cam",
                    rr.Pinhole(resolution=[view['img'].shape[0], view['img'].shape[1]],
                                image_from_camera=view["camera_intrinsics"], 
                                camera_xyz=rr.ViewCoordinates.RDF
                                )
                    )
            
            img = img.numpy().transpose(1,2,0)
            rr.log(f"/world/cam_{i}/points",rr.Points3D(pts3d,colors=img.reshape(-1,3)))
            

            img=img*255

            rr.log(f"world/cam_{i}/cam",rr.Image(img))
        
        if i < 15:
            for j in range(i+1,16):
                rr.log(f"/world/cam_{j}",rr.Clear(recursive=True))

        import pdb; pdb.set_trace()
            
    # for idx in range(len(dataset)):
    #     views = dataset[(idx,0)]
    #     print([view['label'] for view in views])