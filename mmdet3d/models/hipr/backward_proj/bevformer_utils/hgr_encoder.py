# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
import time
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class HGR_encoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, 
                 pc_range=None, 
                 grid_config=None, 
                 data_config=None, 
                 return_intermediate=False, 
                 dataset_type='nuscenes', 
                 fix_bug=False,
                 use_bev_height_mask=None,
                 img_plane=(256,704),
                 **kwargs):

        super(HGR_encoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fix_bug = fix_bug
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.use_bev_height_mask = use_bev_height_mask
        self.img_plane = img_plane


    def get_reference_points(self,H, W, Z=8, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':

            X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1]/2
            Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1]/2
            Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1]/2
            Y, X, Z = torch.meshgrid([Y, X, Z])
            coords = torch.stack([X, Y, Z], dim=-1)
            coords = coords.to(dtype).to(device)
            # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
            return coords

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, cam_params):
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        # ogfH, ogfW = (256, 704)
        ogfH, ogfW = self.img_plane

        if len(reference_points.shape)!=6:
            reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3) 
        ###################################################################
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps) 
                 & (reference_points_cam[..., 0:1] < (1.0-eps)) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]



    # height map -> ref_3d
    def build_ref3d_from_height_map(self, height_map, device=None, dtype=torch.float32, ignore=255):
        """
        输入:
        height_map: (B, X, Y)  —— 每格最高前景体素的 z 索引；忽略格=255
        输出:
        ref3d: (B, Y, X, D, 3) —— 最后一维是 (x, y, z)
                - 有效格: Z 在 [-1, z_top] 线性均匀采样 D 点
                - 忽略格: Z 为整柱范围 [z0,z1] 的均匀中心
        valid: (B, Y, X)      —— 有效格布尔掩码
        """
        assert height_map.dim() == 3, f"height_map must be (B,X,Y), got {height_map.shape}"
        if device is None:
            device = height_map.device
        B, X, Y = height_map.shape

        x0, x1, dx = self.x_bound # [-40, 40, 0.4]
        y0, y1, dy = self.y_bound # [-40, 40, 0.4]
        z0, z1, dz = self.z_bound # [-1, 5.4, 0.8]

        # 轴长度（与 bound 对齐）
        nx = int(round((x1 - x0) / dx))
        ny = int(round((y1 - y0) / dy))
        nz = int(round((z1 - z0) / dz))
        assert X == nx and Y == ny, f"height_map (X,Y)=({X},{Y}) not match bounds ({nx},{ny})"

        # X/Y 网格中心
        xs = torch.arange(x0, x1, dx, device=device, dtype=dtype) + dx * 0.5  # (X,)
        ys = torch.arange(y0, y1, dy, device=device, dtype=dtype) + dy * 0.5  # (Y,)

        Yv, Xv = torch.meshgrid(ys, xs, indexing='ij')  # (Y, X)
        z_centers = z0 + (torch.arange(nz, device=device, dtype=dtype) + 0.5) * dz  # (D,)

       
        # 有效/忽略 与 z_top（把 height_map 从 (B,X,Y) 变到 (B,Y,X)）
        valid_bxy = (height_map != ignore)                                   # (B,X,Y)
        valid = valid_bxy.permute(0, 2, 1).contiguous()                      # (B,Y,X)
        z_top_bxy = height_map                   # (B,X,Y)
        z_top = z_top_bxy.permute(0, 2, 1).contiguous()                      # (B,Y,X)

        # 扩展到 (B,Y,X,D)
        Yv4 = Yv[None, :, :, None].expand(B, ny, nx, nz)     # (B,Y,X,D)
        Xv4 = Xv[None, :, :, None].expand(B, ny, nx, nz)     # (B,Y,X,D)

        z_full = z_centers[None, None, None, :].expand(B, ny, nx, nz)        # (B,Y,X,D)

        base = torch.linspace(-1.0, 1.0, steps=nz, device=device, dtype=dtype)  # (D,)
        base01 = (base + 1.0) / 2.0                                           # [0,1], (D,)
        # z_interp = (-1.0) + base01[None, None, None, :] * (z_top[..., None] + 1.0)  # (B,Y,X,D)
        z_interp = z0 + base01[None, None, None, :] * (z_top[..., None] + 1.0)  # (B,Y,X,D)

        Z4 = torch.where(valid[..., None], z_interp, z_full)   # (B,Y,X,D)

        # 最后一维按 (x, y, z) 堆叠
        ref3d = torch.stack([Xv4, Yv4, Z4], dim=-1)  # (B,Y,X,D,3)

        # 增加cam dim
        ref3d = ref3d[:, None, ...].expand(-1, 6, -1, -1, -1, -1)     # (B, N, Y, X, D, 3)
        return ref3d  


    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                cam_params=None,
                gt_bboxes_3d=None,
                pred_img_depth=None,
                bev_mask=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        _, B, _ = bev_query.shape


        bev_height_map = kwargs.get("bev_height_map", None)       # (B, X, Y)
        bev_height_mask = kwargs.get("bev_height_mask", None)   # (B, X, Y)


        if self.use_bev_height_mask:
            bev_mask = bev_height_mask.permute(0, 2, 1).contiguous().view(B, -1)  # (B, X, Y) -> (B, Y, X) -> (B, Y*X)
        
        ref_3d = self.build_ref3d_from_height_map(bev_height_map)

        ref_2d = self.get_reference_points(bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        ref_3d, reference_points_cam, per_cam_mask_list, bev_query_depth = self.point_sampling(ref_3d, cam_params=cam_params)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        for lid, layer in enumerate(self.layers):
           
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                prev_bev=prev_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                per_cam_mask_list=per_cam_mask_list,
                bev_mask=bev_mask,
                bev_query_depth=bev_query_depth,
                pred_img_depth=pred_img_depth,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class HGR_layer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 selfatten_mask=None,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(HGR_layer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        self.selfatten_mask = selfatten_mask
        assert len(operation_order) in {2, 4, 6}
        # assert set(operation_order) in set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                bev_mask=None,
                bev_query_depth=None,
                per_cam_mask_list=None,
                lidar_bev=None,
                pred_img_depth=None, 
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                if self.selfatten_mask:
                    query = self.attentions[attn_index](
                        query,
                        None,
                        None,
                        identity if self.pre_norm else None,
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=~bev_mask,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        **kwargs)
                    attn_index += 1
                    identity = query
                else:
                    query = self.attentions[attn_index](
                        query,
                        None,
                        None,
                        identity if self.pre_norm else None,
                        query_pos=bev_pos,
                        key_pos=bev_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=None,
                        reference_points=ref_2d,
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        **kwargs)
                    #import pdb;pdb.set_trace()
                    attn_index += 1
                    identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_query_depth=bev_query_depth,
                    pred_img_depth=pred_img_depth,
                    bev_mask=bev_mask,
                    per_cam_mask_list=per_cam_mask_list,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
