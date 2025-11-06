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


class PoseGraphOpt(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses @ node1.Inv() @ node2

        return error.Log().tensor()
    

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