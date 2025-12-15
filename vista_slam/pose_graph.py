import torch
import pypose as pp
import torch.nn as nn

class PoseGraphEdges:
    # all the edges in the pose graph
    def __init__(self, buffer_size:int, device:torch.device):
        self.bs = buffer_size
        self.device = device
        self.reset()
        self.id_pose_conf = 2.0  # confidence for poses among nodes of the same view
    
    def add_edge(self,i,j,se3_ij, conf_ij):
        self.edges[self.num_edges] = torch.tensor([i,j])
        self.poses[self.num_edges] = se3_ij
        self.confs[self.num_edges] = conf_ij
        self.num_edges += 1   

    def reset(self):
        self.poses = pp.identity_Sim3(self.bs).to(self.device)
        self.edges = -torch.ones(self.bs, 2, dtype=torch.int64).to(self.device)
        self.confs = torch.ones(self.bs, 7).to(self.device)
        self.num_edges = 0

class PoseGraphNodes:
    def __init__(self, buffer_size:int, device:torch.device):
        self.bs = buffer_size
        self.device = device
        self.reset()
    
    def add_node(self, view_id, depth, conf, intri, connected_view):
        '''
        add a node to the pose graph. 
        Notice that the absolute pose of the node is not set here, it will be set later
        '''
        n_idx = self.num_nodes
        self.pcl.append((depth.cpu(), conf.cpu(), intri.cpu()))
        self.node_to_view[n_idx] = view_id
        self.node_to_connected_view[n_idx] = connected_view
        self.view_to_node[view_id].append(n_idx)
        mean_conf = conf.mean().item()
        if mean_conf > self.view_to_best_node[view_id][1]:
            self.view_to_best_node[view_id] = (n_idx, mean_conf)
        self.num_nodes += 1
        return n_idx
    
    def reset(self):
        self.poses = pp.identity_Sim3(self.bs).to(self.device)
        self.node_to_view = [-1] * self.bs
        self.node_to_connected_view = [-1] * self.bs
        self.view_to_node = {v:[] for v in range(self.bs)}  #{view_id:[node_id, ...]}
        self.view_to_best_node = {v:(-1,-100) for v in range(self.bs)}  # the best node for each view
        self.pcl = []   #(pcl, conf, intri)
        self.num_nodes = 0


class PoseGraphOptAll(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses @ node1.Inv() @ node2

        return error.Log().tensor()


class PoseGraphOpt(nn.Module):
    def __init__(self, nodes, to_optimize_idxs:torch.Tensor="all"):
        super().__init__()
        device = nodes.device
        if to_optimize_idxs == "all":
            to_optimize_idxs = torch.arange(nodes.shape[0], device=device)

        self.idxs_opt = to_optimize_idxs.long().to(device)
        all_idxs = torch.arange(nodes.shape[0], device=device)
        self.idxs_fix = all_idxs[~torch.isin(all_idxs, self.idxs_opt)]

        self.register_buffer(
            "opt_map",
            -torch.ones(nodes.shape[0], dtype=torch.long, device=device)
        )
        self.register_buffer(
            "fix_map",
            -torch.ones(nodes.shape[0], dtype=torch.long, device=device)
        )
        self.opt_map[self.idxs_opt] = torch.arange(len(self.idxs_opt), device=device)
        self.fix_map[self.idxs_fix] = torch.arange(len(self.idxs_fix), device=device)



        self.nodes_opt = pp.Parameter(nodes[self.idxs_opt]).to(device)
        self.nodes_fixed = nodes[self.idxs_fix]


    def get_nodes(self):
        nodes = pp.identity_Sim3(self.idxs_opt.shape[0] + self.idxs_fix.shape[0]).to(self.nodes_opt.device)
        nodes[self.idxs_opt] = self.nodes_opt.detach().clone()
        nodes[self.idxs_fix] = self.nodes_fixed.detach().clone()
        return nodes

    def forward(self, edges, poses):
        # edges: [E, 2] (global indices)

        in_opt = torch.isin(edges, self.idxs_opt)   # [E, 2]
        both_opt   = in_opt[:, 0] & in_opt[:, 1]
        first_opt  = in_opt[:, 0] & ~in_opt[:, 1]
        second_opt = ~in_opt[:, 0] & in_opt[:, 1]

        # -----------------------------
        # Type 1: both endpoints optimized
        # -----------------------------
        edges_both = edges[both_opt]   # global
        i_both = self.opt_map[edges_both[:, 0]]  # local opt idx
        j_both = self.opt_map[edges_both[:, 1]]  # local opt idx

        nodes_i_both = self.nodes_opt[i_both]
        nodes_j_both = self.nodes_opt[j_both]

        # -----------------------------
        # Type 2: first optimized, second fixed
        # -----------------------------
        edges_first = edges[first_opt]
        i_first = self.opt_map[edges_first[:, 0]]  # opt
        j_first = self.fix_map[edges_first[:, 1]]  # fixed

        nodes_i_first = self.nodes_opt[i_first]
        nodes_j_first = self.nodes_fixed[j_first]

        # -----------------------------
        # Type 3: first fixed, second optimized
        # -----------------------------
        edges_second = edges[second_opt]
        i_second = self.fix_map[edges_second[:, 0]]  # fixed
        j_second = self.opt_map[edges_second[:, 1]]  # opt

        nodes_i_second = self.nodes_fixed[i_second]
        nodes_j_second = self.nodes_opt[j_second]

        error_both = poses[both_opt] @ nodes_i_both.Inv() @ nodes_j_both
        error_first = poses[first_opt] @ nodes_i_first.Inv() @ nodes_j_first
        error_second = poses[second_opt] @ nodes_i_second.Inv() @ nodes_j_second

        error = torch.cat([error_both, error_first, error_second], dim=0)
        
        return error.Log().tensor()


    def get_related_edge_idxs(self, edges:torch.Tensor):
        in_opt = torch.isin(edges, self.idxs_opt)   # [E, 2]
        related_mask   = in_opt[:, 0] | in_opt[:, 1]
        return related_mask


# def cpu_solver(A,b):
#     from scipy.sparse import csr_matrix
#     from scipy.sparse.linalg import spsolve
#     device = A.device
#     A = A.cpu().numpy()
#     b = b.cpu().numpy()
#     A_sparse = csr_matrix(A)
#     x = spsolve(A_sparse, b)            
#     return torch.from_numpy(x).to(device).unsqueeze(-1)

# def sparse_solver(A,b):
#     from cholespy import CholeskySolverF, MatrixType
#     def dense_to_coo(M, eps=1e-12):
#         assert M.is_cuda and M.dim() == 2
#         mask = M.abs() > eps
#         ii, jj = torch.nonzero(mask, as_tuple=True)
#         x = M[ii, jj]
#         return ii.long(), jj.long(), x
#     ii, jj, val = dense_to_coo(A)
#     print(A.shape[0],val.shape/A.shape[0]*A.shape[1])
#     solver = CholeskySolverF(A.shape[0], ii, jj, val, MatrixType.COO)
#     x = torch.zeros_like(b)
#     solver.solve(b, x)
#     return x