import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from .pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from .patch_embed import get_patch_embed

from .blocks.sta_blocks import Block
from .blocks.sta_blocks import DecoderBlock

from .heads import head_factory
from .heads.pose_head import PoseHead_small
from .heads.postprocess import reg_dense_conf

from ..utils.device import MyNvtxRange
from ..utils.misc import freeze_all_params, transpose_to_landscape

inf = float('inf')




class SymmetricTwoViewAssociation(nn.Module):
    """Frontend of ViSTA-SLAM, with the following components:
    - patch embeddings
    - positional embeddings
    - encoder and decoder 
    - downstream heads for 3D point and confidence map prediction
    """
    def __init__(self,
                 img_size=(224,224),           # input image size
                 patch_size=16,          # patch_size 
                 enc_embed_dim=1024,      # encoder feature dimension
                 enc_depth=24,           # encoder depth 
                 enc_num_heads=16,       # encoder number of heads in the transformer block 
                 dec_embed_dim=768,      # decoder feature dimension 
                 dec_depth=12,            # decoder depth 
                 dec_num_heads=12,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pos_embed='RoPE100',     # positional embedding (either cosine or RoPE100)
                 output_mode='pts3d',
                 head_type='dpt',      # 'linear' or 'dpt'
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                ):    

        super().__init__()

        self.img_size = img_size

        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim)
        # positional embeddings in the encoder and decoder
        self._set_pos_embed(pos_embed, enc_embed_dim, dec_embed_dim, 
                            self.patch_embed.num_patches)
        # transformer for the encoder 
        self._set_encoder(enc_embed_dim, enc_depth, enc_num_heads, 
                            mlp_ratio, norm_layer)

        # transformer for the decoder
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, 
                          mlp_ratio, norm_layer)
        # dust3r specific initialization
        self._set_downstream_head(output_mode, head_type, landscape_only, 
                                  depth_mode, conf_mode, patch_size, img_size)
        self.set_freeze(freeze)
        
        self.dec_embed_dim = dec_embed_dim

    def _set_patch_embed(self, patch_embed_cls, img_size=224, patch_size=16, 
                         enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(patch_embed_cls, img_size, 
                                           patch_size, enc_embed_dim)
        
    def _set_encoder(self, enc_embed_dim, enc_depth, enc_num_heads, 
                     mlp_ratio, norm_layer):
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)    
    
    def _set_pos_embed(self, pos_embed, enc_embed_dim, 
                       dec_embed_dim, num_patches):
        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(num_patches**.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, 
                     dec_depth, mlp_ratio, norm_layer):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the two ssymmetric decoders 
        self.dec_block = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=True, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_downstream_head(self, output_mode, head_type, landscape_only, 
                             depth_mode, conf_mode, patch_size, img_size):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head_pts = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head_pts = transpose_to_landscape(self.downstream_head_pts, activate=landscape_only)
        self.head_pose_s = PoseHead_small(input_dim=self.dec_embed_dim)
        self.init_pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)
        


    def set_freeze(self, freeze):  # this is for use by downstream models
        if freeze == 'none':
            return
        if freeze == 'encoder':
            freeze_all_params([self.patch_embed, self.enc_blocks])
        elif freeze == 'corr_score_head_only':
            for param in self.parameters():
                param.requires_grad = False
            for param in self.corr_score_proj.parameters():
                param.requires_grad = True
            for param in self.corr_score_norm.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError(f"freeze={freeze} not implemented")

    def _encode_image(self, image, true_shape, normalize=True):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # add positional embedding without cls token
        if(self.pos_embed != 'cosine'):
            assert self.enc_pos_embed is None 
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)
        if normalize:
            x = self.enc_norm(x)
        return x, pos
    

    def _decode_stereo(self,
                       feat1:torch.Tensor, feat2:torch.Tensor,
                       pose1:torch.Tensor, pose2:torch.Tensor):
        """exchange information between reference and source views in the decoder

        About naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.

        Args:
            ref_feats (B, S, D_enc): img tokens of reference views 
            src_feats (B, S, D_enc): img tokens of source views
            ref_poses (B, S, 2): positions of tokens of reference views
            src_poses (B, S, 2): positions of tokens of source views

        
        Returns:
            final_refs: list of (B, S, D_dec)
            final_srcs: list of (B, S, D_dec)
        """
        # R: number of reference views
        # V: total number of reference and source views
        # S: number of tokens

        
        final_1 = []
        final_2 = []
        # project to decoder dim

        pose_token_1 = self.init_pose_token.expand(feat1.shape[0], -1, -1)
        pose_token_2 = self.init_pose_token.expand(feat2.shape[0], -1, -1)

        f1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)

        f1 = torch.cat([pose_token_1, f1], dim=1)
        f2 = torch.cat([pose_token_2, f2], dim=1)
        pose_pos_1 = -torch.ones(
                pose1.shape[0], 1, 2, device=f1.device, dtype=pose1.dtype)
        pose_pos_2 = -torch.ones(
                pose2.shape[0], 1, 2, device=f2.device, dtype=pose2.dtype)
        pose1 = torch.cat([pose_pos_1, pose1], dim=1)
        pose2 = torch.cat([pose_pos_2, pose2], dim=1)

        final_1.append(f1) 
        final_2.append(f2)
        
        for i in range(self.dec_depth):
            # (R, B, S, D),  (V-R, B, S, D)
            # add pointmap tokens if available(used in Local2WorldModel)
            inputs_1 = final_1[-1]
            inputs_2 = final_2[-1]

            # reference image side
            outputs_1 = self.dec_block[i](inputs_1, inputs_2, 
                                             pose1, pose2) # (B, S, D)
            # source image side
            outputs_2 = self.dec_block[i](inputs_2, inputs_1, 
                                           pose2, pose1) # (B, S, D)
            # store the result
            final_1.append(outputs_1)
            final_2.append(outputs_2)

        # normalize last output
        final_1[-1] = self.dec_norm(final_1[-1])  #(B, S, D)
        final_2[-1] = self.dec_norm(final_2[-1])

        return final_1, final_2  # list of dec_depth+1 elements
    

    def forward(self, views:dict, loop_num=0):
        main_view = views['main_view']
        neighbor_views = views['neighbor_views']
        loop_views_candi = views['loop_views']

        if not self.training:
            loop_num = len(loop_views_candi)
        loop_views = loop_views_candi[:loop_num]
        support_views = neighbor_views + loop_views

        main_enc_feat, main_enc_pos = self._encode_image(main_view['img'],
                                                         main_view['true_shape'],
                                                         normalize=False)
        main_res = []
        support_res = []
        for n_view in support_views:
            n_res={}
            m_res={}
            n_enc_feat, n_enc_pos = self._encode_image(n_view['img'],
                                                       n_view['true_shape'],
                                                       normalize=False)
            main_dec_feat, n_dec_feat = self._decode_stereo(main_enc_feat, n_enc_feat,
                                                            main_enc_pos, n_enc_pos) 

            n_head_pts_input = [n_enc_feat]+[tok[:,1:,:].float() for tok in n_dec_feat]
            n_res_pts = self.head_pts(n_head_pts_input, n_view['true_shape'])
            n_res_pose = self.head_pose_s(n_dec_feat[-1][:,0,:])

            main_head_pts_input = [main_enc_feat]+[tok[:,1:,:].float() for tok in main_dec_feat]
            main_res_pts = self.head_pts(main_head_pts_input, main_view['true_shape'])
            main_res_pose = self.head_pose_s(main_dec_feat[-1][:,0,:])

            n_res['pts3d_pred'] = n_res_pts['pts3d']
            n_res['conf'] = n_res_pts['conf']
            n_res['relative_pose'] = n_res_pose['pose']
            n_res['relative_pose_conf'] = n_res_pose['conf']
            support_res.append(n_res)

            m_res['pts3d_pred'] = main_res_pts['pts3d']
            m_res['conf'] = main_res_pts['conf']
            m_res['relative_pose'] = main_res_pose['pose']
            m_res['relative_pose_conf'] = main_res_pose['conf']
            main_res.append(m_res)
        
        return {'main_views':main_res, 'support_views':support_res}
        