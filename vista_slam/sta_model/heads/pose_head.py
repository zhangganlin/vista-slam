import torch
import torch.nn as nn
import torch.nn.functional as F


# parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
class PoseHead_small(nn.Module):
    """ 
    pose regression head
    """
    def __init__(self, 
                 input_dim, 
                 rot_representation='9D'):
        super().__init__()
        self.rot_representation = rot_representation  
        self.output_dim = 512
        self.input_dim = input_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim,self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim,self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim,self.output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(self.output_dim, 3)

        self.fc_conf = nn.Sequential(
            nn.Linear(self.output_dim, 1),
            nn.Sigmoid()
        )

        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(self.output_dim, 9)
        else:
            self.fc_rot = nn.Linear(self.output_dim, 6)
        
    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

    
    def svd_orthogonalize_stable(self,o, n_iter=100):
        if o.dim() < 3:
            o = o.view(-1, 3, 3)
        o = o / (o.norm(dim=(-2, -1), keepdim=True) + 1e-8)
        for _ in range(n_iter):
            o = (o + torch.linalg.inv(o.transpose(-1, -2))) / 2

        det = torch.det(o)
        o[..., :, -1] *= torch.sign(det).unsqueeze(-1)
        
        return o

    def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    
    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            if self.training:
                out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
            else:
                # out_r = self.svd_orthogonalize_stable(out_r)  # for H100 inference 
                out_r = self.svd_orthogonalize(out_r)
        else:
            out_r = self.rotation_6d_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def forward(self, pose_token):
        B,C = pose_token.shape

        feat = self.mlp(pose_token)  # B,output_dim

        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,9]
        out_conf = self.fc_conf(feat).squeeze(-1)
        pose = self.convert_pose_to_4x4(B, out_r, out_t, pose_token.device)
        res = {"pose": pose, "conf":out_conf}

        return res