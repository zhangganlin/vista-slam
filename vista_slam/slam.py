import torch
import numpy as np
import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
import cv2

from .sta_model.sta_model import SymmetricTwoViewAssociation as STA
from .utils.slam_utils import estimate_intrinsic_from_pts3d, compute_local_pointclouds, estimate_scale_with_depth_and_confidence
from .utils.slam_utils import FontColor, print_msg, suppress_specific_print
from .flow_tracker import FlowTracker
from .pose_graph import PoseGraphEdges, PoseGraphNodes, PoseGraphOpt
from .loop_detector import LoopDetector
import os

import munch
import time

class OnlineSLAM:
    def __init__(self, ckpt_path:str, vocab_path:str, 
                 verbose:bool=False, max_view_num:int=400,
                 neighbor_edge_num:int=3, loop_edge_num:int=3,
                 loop_dist_min:int=40, loop_nms:int=40,
                 loop_cand_thresh_neighbor:int=5,
                 conf_thres:float=4.2, rel_pose_thres:float=0.75, 
                 flow_thres:float=5.0, pgo_every:int=500, live_mode:bool=False):
        self.device = torch.device("cuda")
        self.verbose = verbose
        self.max_view_num = max_view_num
        self.neighbor_edge_num = neighbor_edge_num  # node i connect to all [i-neighbor_edge_num, i+neighbor_edge_num]
        self.loop_edge_num = loop_edge_num      # node i has at most loop_edge_num edges for loop closure
        max_nodes = self.max_view_num*(neighbor_edge_num*2+loop_edge_num) # maximum number of nodes
        scale_edges_num = neighbor_edge_num*2 + loop_edge_num - 1 # edges connected to the first processed node of the view
        pose_edges_num = (neighbor_edge_num*2+loop_edge_num)//2 + 1 # edges between different views
        max_edges = self.max_view_num*(scale_edges_num+pose_edges_num) # maximum number of edges
        
        self.conf_thres = conf_thres
        self.rel_pose_thres = rel_pose_thres # relative pose confidence threshold

        self.pose_graph_nodes = PoseGraphNodes(max_nodes,self.device)
        self.pose_graph_edges = PoseGraphEdges(max_edges, self.device)
        self.pose_graph_solver = ppos.Cholesky()   # ppos.Cholesky() or cpu_solver or sparse_solver
        
        self.frontend:STA = self.load_frontend(ckpt_path)
        total_params = sum(p.numel() for p in self.frontend.parameters())
        if self.verbose: print_msg(f"Total parameters: {total_params}", color=FontColor.INFO)
        
        self.enc_features = []  # output of the encoder of frontend
        self.enc_pos = []     # output of the encoder of frontend
        self.img_shapes = []    # true shape of the input image
        self.imgs = []          # input images
        self.view_names = []  # names of the input images

        self.lc_detector = LoopDetector(vocab_path, loop_dist_min, loop_nms, loop_cand_thresh_neighbor)
        self.image_resolution = (224,224)
        
        self.view_num = 0
        self.pgo_every = pgo_every
        
        self.flow_tracker = FlowTracker(flow_thres)
        
        self.time_dict = {
            "prepare_data": 0.0,
            "encoder": 0.0,
            "decoder": 0.0,
            "lc": 0.0,
            "pgo": 0.0,
            "graph_construction": 0.0,  
        }
        self.live_mode = live_mode

    def reset(self):
        self.enc_features = []
        self.enc_pos = []
        self.img_shapes = []
        self.imgs = []
        self.view_names = []
        self.view_num = 0
        self.pose_graph_nodes.reset()
        self.pose_graph_edges.reset()
        self.flow_tracker.reset()
        self.time_dict = {
            "prepare_data": 0.0,
            "encoder": 0.0,
            "decoder": 0.0,
            "lc": 0.0,
            "pgo": 0.0,
            "graph_construction": 0.0,
        }

    def load_frontend(self, ckpt_path:str):
        frontend = STA()
        checkpoint = torch.load(ckpt_path, 
                                map_location='cpu',weights_only=False)
        
        frontend.load_state_dict(checkpoint['model'],strict=True)

        del checkpoint

        frontend.to(self.device)
        frontend.eval()
        return frontend
    
    def pose_graph_optimize(self):
        print_msg(f"Pose graph optimization (at keyframe {self.view_num}) ...", color=FontColor.PoseGraphOpt)
        if self.live_mode:
            print_msg(f"This may cause latency in live mode, please hold the camera steady if possible.",color=FontColor.PoseGraphOpt)
        node_num = self.pose_graph_nodes.num_nodes
        edge_num = self.pose_graph_edges.num_edges

        graph = PoseGraphOpt(self.pose_graph_nodes.poses[:node_num]).to(self.device)
        solver = self.pose_graph_solver
        strategy = ppost.TrustRegion(radius=1e4)
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, vectorize=True)
        scheduler = StopOnPlateau(optimizer, steps=20, patience=3, decreasing=1e-4, verbose=self.verbose)
        weight = torch.diag_embed(self.pose_graph_edges.confs[:edge_num])

        with suppress_specific_print("Linear solver failed. Breaking optimization step...", color=FontColor.PoseGraphOpt):
            scheduler.optimize(input=(self.pose_graph_edges.edges[:edge_num],
                                      self.pose_graph_edges.poses[:edge_num]),
                              weight=weight)
        self.pose_graph_nodes.poses[:node_num] = graph.nodes.detach()
        print_msg("Pose graph optimization done.", color=FontColor.PoseGraphOpt)

    def add_view(self, image, image_shape, view_name):
        with torch.no_grad():
            enc_feat, enc_pos = self.frontend._encode_image(image, image_shape, normalize=False)
        self.enc_features.append(enc_feat)
        self.enc_pos.append(enc_pos)
        self.img_shapes.append(image_shape)
        self.imgs.append(image.squeeze(0).cpu())
        self.view_names.append(view_name)
        self.view_num += 1
        assert len(self.enc_features) == len(self.enc_pos) == len(self.img_shapes) == len(self.imgs) == len(self.view_names) == self.view_num
    
    def regress_two_views(self, i, j):
        decoder_start = time.time()
        
        enc_feat_j = self.enc_features[j]
        enc_feat_i = self.enc_features[i]
        enc_pos_j = self.enc_pos[j]
        enc_pos_i = self.enc_pos[i]

        with torch.no_grad():
            dec_feat_ij, dec_feat_ji = self.frontend._decode_stereo(
                enc_feat_i, enc_feat_j, enc_pos_i, enc_pos_j)
            pose_ij = self.frontend.head_pose_s(dec_feat_ij[-1][:,0,:]) 
        se3_ij = pp.mat2SE3(pose_ij['pose'],atol=1e-3).data
        rel_pose_conf_ij = pose_ij['conf']
        
        if rel_pose_conf_ij < self.rel_pose_thres and i-j!=1:
            return se3_ij, rel_pose_conf_ij, None, None, None

        true_shape_j = self.img_shapes[j]
        true_shape_i = self.img_shapes[i]

        with torch.no_grad():
            ji_head_pts_input = [enc_feat_j]+[tok[:,1:,:].float() for tok in dec_feat_ji]
            ij_head_pts_input = [enc_feat_i]+[tok[:,1:,:].float() for tok in dec_feat_ij]

            ji_ret = self.frontend.head_pts(ji_head_pts_input, true_shape_j)
            ij_ret = self.frontend.head_pts(ij_head_pts_input, true_shape_i)
        
        pcls = torch.cat([ij_ret['pts3d'],ji_ret['pts3d']],dim=0)                #[2,224,224,3]
        confs = torch.cat([ij_ret['conf'],ji_ret['conf']],dim=0)                 #[2,224,224]
        intri = estimate_intrinsic_from_pts3d(pcls,confs,shared_intrinsic=True)  #[3,3]
        depths = pcls[...,2]
        
        decoder_end = time.time()
        self.time_dict['decoder'] += (decoder_end - decoder_start)
        return se3_ij, rel_pose_conf_ij, confs, intri, depths
        
    def connect_view_i_j(self, i, j):
        assert i > j, f"i should be larger than j, but got i={i}, j={j}"
        
        se3_ij, rel_pose_conf_ij, confs, intri, depths = self.regress_two_views(i, j)
        if rel_pose_conf_ij < self.rel_pose_thres and i-j!=1:
            if self.verbose: print_msg(f"Rejecting edge (view {i} [{self.view_names[i]}] -- view {j} [{self.view_names[j]}]) with conf {rel_pose_conf_ij.item():.3f}",
                                   color=FontColor.EdgeReject)
            return False        
        if i-j > self.neighbor_edge_num and self.verbose:
            print_msg(f"Adding loop closure edge (view {i} [{self.view_names[i]}] -- view {j} [{self.view_names[j]}]) with conf {rel_pose_conf_ij.item():.3f}",
                      color=FontColor.LoopClosure)
        
        scale_ij = 1.0
        sim3_ij = pp.Sim3(torch.cat([se3_ij, torch.tensor([[scale_ij]]).to(self.device)], dim=1)).to(self.device)

        node_idx_of_ij = {i:None, j:None}
        view_i_is_new = True
        for v, depth, pcl_conf, v_other in zip([i,j], depths, confs,[j,i]):
            # adding the node to the pose graph 
            n_idx = self.pose_graph_nodes.add_node(
                            view_id=v, depth=depth, conf=pcl_conf, 
                            intri=intri, connected_view=v_other)
            node_idx_of_ij[v] = n_idx
            
            # adding scale_edge connecting the first processed node of the view and the new node
            if len(self.pose_graph_nodes.view_to_node[v]) > 1:
                if v==i: view_i_is_new = False                
                n_idx_other = self.pose_graph_nodes.view_to_node[v][0]
                depth_other, pcl_conf_other, _ = self.pose_graph_nodes.pcl[n_idx_other]
                depth_other = depth_other.to(self.device)
                pcl_conf_other = pcl_conf_other.to(self.device)
                scale = estimate_scale_with_depth_and_confidence(
                    depth,depth_other, pcl_conf, pcl_conf_other).to(self.device)
                scale = scale.unsqueeze(0).unsqueeze(0)  # [1,1,3]
                scale_conf = (pcl_conf* pcl_conf_other).sqrt().mean().item()
                edge_weight = torch.tensor([self.pose_graph_edges.id_pose_conf]*6+[scale_conf]).to(self.device)
                se3_id = pp.identity_SE3(1).data.to(self.device)
                sim3 = pp.Sim3(torch.cat([se3_id,scale], dim=1)).to(self.device)
                self.pose_graph_edges.add_edge(n_idx,n_idx_other, sim3, edge_weight)
                self.pose_graph_nodes.poses[n_idx] = self.pose_graph_nodes.poses[n_idx_other] @ sim3

        if view_i_is_new:  # only view i could be new, since j < i, which means j is already in the graph
            pose_j = self.pose_graph_nodes.poses[node_idx_of_ij[j]]
            self.pose_graph_nodes.poses[node_idx_of_ij[i]] = pose_j @ sim3_ij
        
        # adding the pose_edge between the two nodes
        self.pose_graph_edges.add_edge(node_idx_of_ij[i], node_idx_of_ij[j],sim3_ij, rel_pose_conf_ij)

        return True


    def step(self, value, force_pgo=False, log_intermediate_results=False, output_folder=None):
        prepare_start = time.time()
        image = value['rgb']
        image_shape = value['shape']
        img_gray = value['gray']
        view_name = value['view_name']
        i = self.view_num
        if i == 0:
            sim3 = pp.identity_Sim3(1).to(self.device)
            self.pose_graph_nodes.poses[0] = sim3
        prepare_end = time.time()
        self.time_dict['prepare_data'] += (prepare_end - prepare_start)
        
        enc_start = time.time()
        self.add_view(image, image_shape, view_name)
        enc_end = time.time()
        self.time_dict['encoder'] += (enc_end - enc_start)
        
        graph_neighbor_start = time.time()
        farthest_neighbor = max(0,i-self.neighbor_edge_num)
        for j in range(farthest_neighbor, i):
            self.connect_view_i_j(i,j)
        graph_neighbor_end = time.time()

        loop_start = time.time()
        loop_candi_list = self.lc_detector.detect_loop(img_gray, farthest_neighbor)
        loop_end = time.time()
        self.time_dict['lc'] += (loop_end - loop_start)

        # check and add top loop candidates
        graph_loop_start = time.time()        
        for j_sim in loop_candi_list[:self.loop_edge_num]:
            j = j_sim[0]
            self.connect_view_i_j(i,j)
        graph_loop_end = time.time()
        self.time_dict['graph_construction'] += (graph_neighbor_end - graph_neighbor_start) + (graph_loop_end - graph_loop_start)
        
        
        if self.view_num % self.pgo_every == 0 or force_pgo:
            if log_intermediate_results:
                self.save_data_all(f"{output_folder}",
                        save_view_graph=False, traj_name_postfix=f"{self.view_num-1}",
                        save_poses=True, save_images=False, save_scales=True,
                        save_depths=False, save_intrinsics=False, 
                        save_confs=False, save_ply=False,
                        gt_poses=None, gt_depths=None, gt_intrinsics=None)
            opt_start = time.time()
            self.pose_graph_optimize()
            opt_end = time.time()
            self.time_dict['pgo'] += (opt_end - opt_start)
            
            torch.cuda.empty_cache()
            return True
        return False
    
    def get_view(self,v, filter_outlier=True, 
                 return_pose=True, return_depth=True, return_intri=True):
        view = {}
        
        best_node = self.pose_graph_nodes.view_to_best_node[v][0]
        Sim3 = self.pose_graph_nodes.poses[best_node]
        if return_pose:
            rot = Sim3.rotation().matrix().cpu()  # 3,3
            trans = Sim3.translation().cpu()  # 3,1
            pose = torch.eye(4,dtype=rot.dtype,device=rot.device)
            pose[:3,:3] = rot
            pose[:3,3] = trans
            view['pose'] = pose.cpu()
        
        if return_depth:
            scale = Sim3.scale().cpu()  # 1,1
            depth = self.pose_graph_nodes.pcl[best_node][0] * scale
            conf = self.pose_graph_nodes.pcl[best_node][1]
            mask = conf<self.conf_thres
            if filter_outlier:
                depth[mask] = 0.0
            view['depth'] = depth.cpu()
        
        if return_intri:
            intri = self.pose_graph_nodes.pcl[best_node][2]
            view['intri'] = intri.cpu()
        
        return munch.Munch(view)
        
    def get_view_graph(self):
        view_graph = {}
        for v in range(self.view_num):
            neighbors = []
            for u in self.pose_graph_nodes.view_to_node[v]:
                neighbor_view = self.pose_graph_nodes.node_to_connected_view[u]
                neighbors.append(neighbor_view)
            view_graph[v] = neighbors
        return view_graph

    def save_data_all(self, output_folder, 
                  save_view_graph=True, traj_name_postfix=None,
                  save_poses=True, save_images=True, save_scales=True,
                  save_depths=True, save_intrinsics=True, 
                  save_confs=True, save_ply=True, 
                  gt_poses=None, gt_depths=None, gt_intrinsics=None
                  ):        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if save_view_graph:
            view_graph = self.get_view_graph()
            view_graph_save_path = f"{output_folder}/view_graph.npz" 
            np.savez(view_graph_save_path, 
                     view_graph=view_graph, loop_min_dist=self.lc_detector.loop_dist_min,
                     view_names=self.view_names)
        poses = []
        depths = []
        scales = []
        confs = []
        intrinsics = []
        for v in range(self.view_num):
            best_node = self.pose_graph_nodes.view_to_best_node[v][0]
            Sim3 = self.pose_graph_nodes.poses[best_node]
            rot = Sim3.rotation().matrix().cpu()  # 3,3
            trans = Sim3.translation().cpu()  # 3,1
            scale = Sim3.scale().cpu()  # 1,1
            depth = self.pose_graph_nodes.pcl[best_node][0]
            conf = self.pose_graph_nodes.pcl[best_node][1]
            intri = self.pose_graph_nodes.pcl[best_node][2]
            pose = torch.eye(4,dtype=rot.dtype,device=rot.device)
            pose[:3,:3] = rot
            pose[:3,3] = trans
            poses.append(pose.cpu())
            depths.append(depth.cpu())
            scales.append(scale.cpu())
            confs.append(conf.cpu())
            intrinsics.append(intri.cpu())
        poses = torch.stack(poses, dim=0)  # N,4,4
        depths = torch.stack(depths, dim=0)  # N,H,W
        scales = torch.stack(scales, dim=0)  # N,1
        confs = torch.stack(confs, dim=0)  # N,H,W
        intrinsics = torch.stack(intrinsics, dim=0)  # N,3,3
        masks = confs > self.conf_thres
        
        images = torch.stack(self.imgs, dim=0).cpu().float().permute(0, 2, 3, 1)  # N,H,W,3
        images = (images + 1.0) / 2.0  # normalize to [0,1]
        traj_postfix = f"_{traj_name_postfix}" if traj_name_postfix is not None else ""
        if save_poses:
            np.save(f"{output_folder}/trajectory{traj_postfix}.npy" , poses.cpu().numpy())
        if save_scales:
            np.save(f"{output_folder}/scales{traj_postfix}.npy", scales.cpu().numpy())
        if save_images:
            np.save(f"{output_folder}/images.npy", images.numpy()) 
        if save_depths:
            np.save(f"{output_folder}/depths.npy", depths.cpu().numpy())
        if save_confs:
            np.savez(f"{output_folder}/confs.npz", confs=confs.cpu().numpy(),thres=self.conf_thres)           
        if save_intrinsics:
            np.save(f"{output_folder}/intrinsics.npy", intrinsics.cpu().numpy())
        if save_ply:
            import open3d as o3d
            scaled_depths = depths * scales.unsqueeze(-1)
            local_points = compute_local_pointclouds(scaled_depths, intrinsics)
            N,H,W, _ = local_points.shape
            local_points_flat = local_points.view(N, -1, 3)
            ones = torch.ones(N, local_points_flat.shape[1], 1, device=local_points.device)
            points_hom = torch.cat([local_points_flat, ones], dim=-1)  # (N, H*W, 4)
            world_points_hom = torch.bmm(points_hom.float(), poses.transpose(1, 2).float())  # (N, H*W, 4)
            world_points = world_points_hom[..., :3].view(N, H, W, 3)
            points = world_points[masks].cpu().numpy()
            colors = images[masks].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(f"{output_folder}/pointcloud.ply", pcd)
        
        if gt_poses is not None:
            gt_poses = np.array(gt_poses).astype(np.float32)
            np.save(f"{output_folder}/gt_poses.npy", gt_poses)
        if gt_depths is not None:
            gt_depths = np.array(gt_depths).astype(np.float32)
            np.save(f"{output_folder}/gt_depths.npy", gt_depths)
        if gt_intrinsics is not None:
            np.save(f"{output_folder}/gt_intrinsics.npy", gt_intrinsics)    
            
    def get_pointmap_vis(self,v):
        view = self.get_view(v, filter_outlier=False, return_pose=False, return_depth=True, return_intri=True)
        depth, intri = view.depth, view.intri        
        pcl = compute_local_pointclouds(depth.unsqueeze(0), intri).squeeze(0)
        pcl = pcl.cpu().numpy()
        min_vals = pcl.min(axis=(0,1), keepdims=True)
        max_vals = pcl.max(axis=(0,1), keepdims=True)
        normalized = (pcl - min_vals) / (max_vals - min_vals + 1e-8)  # avoid div-by-zero
        img = (normalized * 255).astype(np.uint8)
        return img, pcl
        
    def save_pointmap(self, v, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        img, pcl = self.get_pointmap_vis(v)
        np.save(f"{output_folder}/pointmap_cam_{v}.npy", pcl)
        # save as PNG (keeps RGB)
        cv2.imwrite(f"{output_folder}/pointmap_cam_{v}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def get_time_dict(self):
        time_dict = self.time_dict.copy()
        time_dict['graph_construction'] = time_dict['graph_construction'] - time_dict['decoder']  # decoder is part of graph construction
        total_time = sum(time_dict.values())
        time_dict['total'] = total_time
        return time_dict