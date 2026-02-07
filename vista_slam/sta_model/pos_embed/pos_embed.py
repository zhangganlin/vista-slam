# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

try:
    from .curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError as e:
    # print(e)
    print(f'Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self._inv_freq_cache = {}
            self._sin_cos_cache = {}
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def _get_inv_freq(self, q, device, dtype):
            key = (device.type, device.index, dtype, q)
            inv_freq = self._inv_freq_cache.get(key)
            if inv_freq is None:
                inv_freq = self.F0 / (self.base ** (torch.arange(q, device=device, dtype=dtype) / q))
                self._inv_freq_cache[key] = inv_freq
            return inv_freq

        def _get_sin_cos_cache(self, min_pos, max_pos, q, device, dtype):
            key = (device.type, device.index, dtype, q)
            cached = self._sin_cos_cache.get(key)
            if cached is None or cached[0] > min_pos or cached[1] < max_pos:
                inv_freq = self._get_inv_freq(q, device, dtype)
                positions = torch.arange(min_pos, max_pos + 1, device=device, dtype=dtype)
                freqs = positions[:, None] * inv_freq[None, :]
                cos = freqs.cos()
                sin = freqs.sin()
                self._sin_cos_cache[key] = (min_pos, max_pos, cos, sin)
            else:
                _, _, cos, sin = cached
            return cos, sin
            
        def apply_rope1d(self, tokens, pos1d):
            assert pos1d.ndim == 2
            q = tokens.size(-1) // 2
            if pos1d.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                pos_index = pos1d.to(torch.long)
                min_pos = int(pos_index.min().item())
                max_pos = int(pos_index.max().item())
                cos_cache, sin_cache = self._get_sin_cos_cache(min_pos, max_pos, q, tokens.device, tokens.dtype)
                offset = -min_pos
                cos = cos_cache[pos_index + offset].unsqueeze(1)
                sin = sin_cache[pos_index + offset].unsqueeze(1)
            else:
                inv_freq = self._get_inv_freq(q, tokens.device, tokens.dtype)
                freqs = pos1d.to(tokens.dtype).unsqueeze(-1) * inv_freq
                cos = freqs.cos().unsqueeze(1)
                sin = freqs.sin().unsqueeze(1)
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0])
            x = self.apply_rope1d(x, positions[:, :, 1])
            tokens = torch.cat((y, x), dim=-1)
            return tokens