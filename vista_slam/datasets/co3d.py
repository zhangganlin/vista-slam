import os.path as osp
import os
import cv2
import numpy as np
import math
import glob
import tqdm
import torch
from .base.base_view_graph_dataset import BaseViewGraphDataset, is_good_type, transpose_to_landscape
from ..utils.image import imread_cv2   
from ..utils.geometry import depthmap_to_camera_coordinates

class Co3d(BaseViewGraphDataset):
    TRAINING_CATEGORIES = [
        "apple","backpack","banana","baseballbat","baseballglove","bench","bicycle",
        "bottle","bowl","broccoli","cake","car","carrot","cellphone","chair","cup","donut","hairdryer","handbag","hydrant","keyboard",
        "laptop","microwave","motorcycle","mouse","orange","parkingmeter","pizza","plant","stopsign","teddybear","toaster","toilet",
        "toybus","toyplane","toytrain","toytruck","tv","umbrella","vase","wineglass",
    ]
    TEST_CATEGORIES = ["ball", "book", "couch", "frisbee", "hotdog", "kite", "remote", "sandwich", "skateboard", "suitcase"]
    def __init__(self,  
                 sensor_data_root='/data/yuzheng/data/scannetpp/train_val_scannetpp', 
                 scene_name=None,
                 neighbor_num=5,     # sample neighbor_num from [i-neighbor_range, i] and neighbor_num [i, i+neighbor_range]
                 loop_num=5,         # sample loop_num from the loop candidates
                 degree_range=90,  # degree range to select views
                 num_sample_per_scene=10,
                 mask_bg='rand',
                 *args,**kwargs):
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.sensor_data_root = sensor_data_root
        
        self.neighbor_total_num = neighbor_num*2 + loop_num
        self.loop_num = loop_num
        self.mask_bg = mask_bg
        self.num_sample_per_scene = num_sample_per_scene
        self.neighbor_range = int(degree_range / 360 * 200)
        if self.split == 'train':
            self.categories = self.TRAINING_CATEGORIES
        elif self.split=='test':
            self.categories = self.TEST_CATEGORIES
        self.scene_names = []
        for cate in self.categories:
            path = osp.join(sensor_data_root, cate)
            scenes = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
            scene_names =[f"{cate}/{scene}" for scene in scenes]
            self.scene_names += scene_names
        self.valid = {scene_name:None for scene_name in self.scene_names}
        self.view_names = {scene_name:None for scene_name in self.scene_names}

        if scene_name is not None:
            assert self.split is None
            if isinstance(scene_name, list): 
                self.scene_names = scene_name
            else:
                assert isinstance(scene_name, str)
                self.scene_names = [scene_name]
        print(self)


    def __len__(self):
        return len(self.scene_names)*self.num_sample_per_scene
    
    def _read_view(self, data_dir, view_name, resolution, mask_bg, rng):
        try:
            camera_info = np.load(osp.join(data_dir, f"images/{view_name}.npz"))
            intri = camera_info['camera_intrinsics'].astype(np.float32)
            if not np.isfinite(intri).all():
                print(data_dir,view_name, "intri invalid", force=True)
                return False, None
            camera_pose = camera_info['camera_pose'].astype(np.float32)
            if not np.isfinite(camera_pose).all():
                print(data_dir,view_name, "pose invalid", force=True)
                return False, None
            # Load RGB image
            rgb_image = imread_cv2(osp.join(data_dir, f"images/{view_name}.jpg"))
            if not np.isfinite(rgb_image).all():
                print(data_dir,view_name, "rgb invalid", force=True)
                return False, None
            if mask_bg:
                maskpath = osp.join(data_dir, f"masks/{view_name}.png")
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
            # Load depthmap
            depthmap = imread_cv2(osp.join(data_dir, f"depths/{view_name}.jpg.geometric.png"), cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(camera_info['maximum_depth'])
        except:
            print(data_dir,view_name, "load failed", force=True)
            return False, None
        depthmap[~np.isfinite(depthmap)] = 0  # invalid
        if mask_bg:
            depthmap *= maskmap

        rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

        rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            rgb_image, depthmap, intri, resolution,
            w_edge=0, h_edge=0,
            rng=rng, info=f"{data_dir}/{view_name}")

        view = {}

        width, height = rgb_image.size
        view['true_shape'] = np.int32((height, width))
        view['img'] = self.transform(rgb_image)    
        view['camera_pose'] = camera_pose
        view['camera_intrinsics'] = intrinsics
        view['depthmap'] = depthmap
        pts3d_cam, valid_mask = depthmap_to_camera_coordinates(view['depthmap'], view['camera_intrinsics'])
        view['pts3d_cam'] = pts3d_cam  #(H, W, 3)
        
        view['valid_mask'] = valid_mask & np.isfinite(pts3d_cam).all(axis=-1)
        
        valid_gt_pcds = pts3d_cam * view['valid_mask'][..., None]  # (H, W, 3)   
        # Compute scales
        pcd_norm = np.linalg.norm(valid_gt_pcds, axis=-1).sum(axis=(0, 1))

        if view['valid_mask'].sum() == 0 or float(pcd_norm) < 1e-5:
            # print(data_dir,view_name, "depth invalid", force=True)
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

    def sample_frames(self, data_dir, scene_id, resolution, rng, mask_bg):
        img_indices = list(range(len(self.view_names[scene_id])))
        center_view_idx_candidates = img_indices[self.neighbor_range:-self.neighbor_range] 
        success = False
        while not success:
            center_view_idx_candidates = [idx for idx in center_view_idx_candidates
                                          if self.valid[scene_id][idx]]
            if len(center_view_idx_candidates) == 0:
                return None, None
            center_view_idx = rng.choice(center_view_idx_candidates)
            center_view_name = self.view_names[scene_id][center_view_idx]
            success, center_view = self._read_view(data_dir, center_view_name, resolution, mask_bg, rng)
            if not success:
                self.valid[scene_id][center_view_idx] = False
        
        neighbors = []
        neighbors_idx_candidates = img_indices[center_view_idx-self.neighbor_range:center_view_idx+self.neighbor_range+1]
        while len(neighbors) < self.neighbor_total_num:
            neighbors_idx_candidates = [idx for idx in neighbors_idx_candidates 
                                        if self.valid[scene_id][idx] and idx != center_view_idx]
            if len(neighbors_idx_candidates) == 0:
                return None, None
            neighbor_idx = rng.choice(neighbors_idx_candidates)
            neighbor_view_name = self.view_names[scene_id][neighbor_idx]
            success, neighbor_view = self._read_view(data_dir, neighbor_view_name, resolution, mask_bg, rng)
            if not success:
                self.valid[scene_id][neighbor_idx] = False
                continue
            neighbors.append(neighbor_view)

        return center_view, neighbors
  
    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_names[idx // self.num_sample_per_scene]
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
        data_path = osp.join(self.sensor_data_root, scene_id)

        if self.view_names[scene_id] is None:
            self.view_names[scene_id] = sorted(glob.glob(osp.join(data_path, "images/*.jpg")))
            self.view_names[scene_id] = [osp.basename(name).split('.')[0] for name in self.view_names[scene_id]]
            self.valid[scene_id] = [True] * len(self.view_names[scene_id])
        center_frame, neighbors = self.sample_frames(data_path, scene_id, 
                                                     resolution, rng, mask_bg)
        if center_frame is None or neighbors is None:
            print(f"Failed to sample frames for {scene_id}",force=True)

            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)

        views = {'main_view': center_frame, 
                 'neighbor_views': neighbors[self.loop_num:],
                 'loop_views': neighbors[:self.loop_num]}
        return views  
    



# for testing

if __name__ == "__main__":


    def print_for_debug():
        import builtins
        import datetime
        """
        This function disables printing when not in master process
        """
        builtin_print = builtins.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if force:
                now = datetime.datetime.now().time()
                builtin_print('[{}] '.format(now), end='')  # print with time stamp
                builtin_print(*args, **kwargs)

        builtins.print = print

    print_for_debug()
    import rerun as rr

    dataset = Co3d(split="train", resolution=(224,224), 
                      sensor_data_root='/home/wiss/zga/storage/user/datasets/spann3r_preprocess/co3d', 
                      scene_name=None)

    rr.init("scannet_test",spawn=False)
    # rr.connect_tcp("131.159.19.231:9876")
    rr.connect_tcp("0.0.0.0:9876")


    rr.log(f"/world",rr.Transform3D())
    for idx in np.random.permutation(len(dataset))[:10]:
    # for idx in range(len(dataset))[5:10000:2000]:
        views_dict = dataset[(idx,0)]
        rr.set_time_sequence("sample",idx)
        views = [views_dict['main_view']] + views_dict['neighbor_views'] + views_dict['loop_views']

        for i, view in enumerate(views):
            img = view['img']
            img = (img+1.0)/2.0
            
            
            pose = view['camera_pose']
                        
            rr.log(f"/world/cam_{i}",rr.Transform3D(translation=pose[:3,3], mat3x3=pose[:3,:3]))
            rr.log(f"/world/cam_{i}/cam",
                    rr.Pinhole(resolution=[view['img'].shape[0], view['img'].shape[1]],
                                image_from_camera=view["camera_intrinsics"], 
                                camera_xyz=rr.ViewCoordinates.RDF
                                )
                    )
            
            img = img.numpy().transpose(1,2,0)

            
            mask= view['valid_mask']
            pts3d = np.array(view['pts3d_cam'][mask]).reshape(-1,3)
            colors = img[mask].reshape(-1,3)
            rr.log(f"/world/cam_{i}/points",rr.Points3D(pts3d,colors=colors))
            

            img=img*255

            rr.log(f"world/cam_{i}/cam",rr.Image(img))
        
        if i < 15:
            for j in range(i+1,16):
                rr.log(f"/world/cam_{j}",rr.Clear(recursive=True))

            
            
    # for idx in range(len(dataset)):
    #     views = dataset[(idx,0)]
    #     print([view['label'] for view in views])