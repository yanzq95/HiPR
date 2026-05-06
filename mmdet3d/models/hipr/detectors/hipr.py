
import torch
import torch.nn.functional as F
import torch.nn as nn
import mmcv
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.alocc.detectors import ALOCC
import numpy as np
import torch
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.models.backbones.swin_bev import SwinTransformerBEVFT
from mmdet3d.models.backbones.flash_intern_image import FlashInternImage
from mmdet3d.models.alocc.heads.occ_loss_utils import CustomFocalLoss
from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
from mmdet3d.models.alocc.modules.temporal_fusion import VoxelevelHistoryFusion, SceneLevelHistoryFusion, MotionHisoryFusion
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import copy
import math
import os
CNT = 0

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res
def positional_encoding_continual_1d(d_model, position):
    """
    :param d_model: dimension of the token
    :param position: position
    :return: (position, d_model) position embedding matrix
    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros((*position.shape,d_model)).to(position.device)
    # position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))).to(position.device)

    pe[..., 0::2] = torch.sin(position.unsqueeze(-1).float() * div_term)
    pe[..., 1::2] = torch.cos(position.unsqueeze(-1).float() * div_term)

    return pe

@DETECTORS.register_module()
class HiPR(ALOCC):
    def __init__(self, 
                fuse=dict(type="Add"),
                train_iter=0,
                gt_period_end=0,
                mix_period_end=0,
                height_denoise_rate=1.0,
                curr_mode='replace', #  replace
                **kwargs):
        super(HiPR, self).__init__(**kwargs)

        self.fuse = builder.build_neck(fuse) if fuse else None
        self.train_iter = train_iter
        self.gt_period_end = gt_period_end
        self.mix_period_end = mix_period_end
        self.height_denoise_rate = height_denoise_rate
        self.curr_mode = curr_mode

    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        
        return_map = {}
        if self.pose_add_noise:
            rot=img[1]
            tran=img[2]
            img[1]=img[1]+torch.randn_like(img[1])*self.pose_add_noise
            img[2]=img[2]+torch.randn_like(img[2])*self.pose_add_noise
        cam_params = img[1:7]
        
        if self.depth_stereo:
        
            inputs_=[img[0],*kwargs['aux_cam_params']] if self.training else [img[0],*kwargs['aux_cam_params'][0]]
            inputs_stereo_=kwargs['adj_aux_cam_params'] if self.training else kwargs['adj_aux_cam_params'][0]
            imgs, curr2adjsensor = self.prepare_inputs(inputs_,inputs_stereo_)
            context, stereo_feat = self.image_encoder(imgs[0], stereo=True)
            if self.save_stereo and not self.training:
                """
                The problem of different data augmentation in different training epochs has not been solved yet.
                """
                start_of_sequence = torch.BoolTensor([
                    single_img_metas['start_of_sequence'] 
                    for single_img_metas in img_metas]).to(stereo_feat.device)
                if self.history_stereo is None:
                    self.history_stereo=stereo_feat
                elif start_of_sequence.sum()>0:
                    
                    batch_size=imgs[0].shape[0]
                    tem=stereo_feat.reshape(batch_size,stereo_feat.shape[0]//batch_size,*stereo_feat.shape[1:])[start_of_sequence].detach()
                    self.history_stereo=self.history_stereo.reshape(batch_size,self.history_stereo.shape[0]//batch_size,*self.history_stereo.shape[1:])
                    self.history_stereo[start_of_sequence]=tem
                    self.history_stereo=self.history_stereo.reshape(*stereo_feat.shape)
                
                feat_prev_iv=self.history_stereo
                self.history_stereo=stereo_feat
                
            else:
                with torch.no_grad():
                    feat_prev_iv = self.extract_stereo_ref_feat(imgs[1])
               
            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                     intrins=img[3],
                     post_rots=img[4],
                     post_trans=img[5],
                    #  frustum=self.cv_frustum.to(stereo_feat.device),
                     cv_downsample=4,
                     downsample=self.downsample,
                     grid_config=self.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
            #################################################################
       
            depth_occ_volumn=None

        else:
            context,_ = self.image_encoder(img[0])
            stereo_metas = None
            depth_occ_volumn=None
        if  self.learnable_pose:
             
            b,n,c,h,w=context.shape
            context_=context.reshape(b*n,c,h,w)
            rot=cam_params[0].reshape(b*n,-1)
            tran=cam_params[1].reshape(b*n,-1)
            
            pose=self.pose_net(context_)
            pose=pose.mean((2,3))
            pose=self.pose_head(torch.cat((pose,rot,tran),dim=-1))*self.pose_weight
            
            rot=rot+pose[:,:3*3]
            tran=tran+pose[:,3*3:]
            
            rot=rot.reshape(b,n,3,3)
            tran=tran.reshape(b,n,3)
            cam_params[0]=rot
            cam_params[1]=tran
        
            
        return_map['context_before_depth_net']=context
        if not self.depth_stereo and ( self.geometry_his_fusion):
            inputs_=[img[0],*kwargs['aux_cam_params']] if self.training else [img[0],*kwargs['aux_cam_params'][0]]
            inputs_stereo_=kwargs['adj_aux_cam_params'] if self.training else kwargs['adj_aux_cam_params'][0]
            imgs, curr2adjsensor = self.prepare_inputs(inputs_,inputs_stereo_)
            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                     intrins=img[3],
                     post_rots=img[4],
                     post_trans=img[5],
                    #  frustum=self.cv_frustum.to(stereo_feat.device),
                     cv_downsample=4,
                     downsample=self.downsample,
                     grid_config=self.grid_config)
        
           
        if self.with_specific_component('depth_net'):
    
            depth_output = self.depth_net(context, cam_params,stereo_metas,img_metas=img_metas,cost_volumn=depth_occ_volumn,**kwargs)
            context, depth_pred,geometry=depth_output['context'],depth_output['depth_pred'],depth_output['geometry']
            if self.soft_filling_with_offset:
                coor_offsets= depth_output['coor_offsets']

            return_map['depth'] = depth_pred
            return_map['context'] = context
            
                
        else:
            # context=None
            geometry=None
            depth_output=None
        if self.with_specific_component('context_net'):
            mlp_input = self.context_net.get_mlp_input(*cam_params)
            context= self.context_net(context, mlp_input)
        
        if self.with_specific_component('view_transformer'):
            if not self.soft_filling_with_offset:
                coor_offsets=None
            
            if self.load_sem_gt :
                if self.training:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                else:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'][0],kwargs['gt_semantic_map'][0])
                kwargs['gt_imgseg']=gt_imgseg
                kwargs['gt_depth_disc']=gt_depth
            bev_feat = self.view_transformer(cam_params, context, geometry,coor_offsets,depth_output=depth_output, **kwargs) 
            
            return_map['cam_params'] = cam_params
        else:
            bev_feat = None


        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('pre_process'):
            bev_feat = self.pre_process(bev_feat)[0]
        return_map['bev_feat_before_encoder']=bev_feat

        feat_for_pred_bias=None
        if self.scene_his_fusion and self.scene_his_before_vox_his:
                
                start_of_sequence = torch.BoolTensor([
                        single_img_metas['start_of_sequence'] 
                        for single_img_metas in img_metas]).to(img[0][0].device)
                bev_shape=bev_feat.shape[2:]
                bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],-1)
                bev_feat=self.scene_his_func_before_vox_his(bev_feat,start_of_sequence)
                
                bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],*bev_shape)
        if not self.not_use_history:
            if self.use_vox_his_func:
                bev_feat = self.fuse_history_func.forward_fuse_his(bev_feat, img_metas, img[6])['fused_bev_feat']
            else:
                bev_feat = self.fuse_history(bev_feat, img_metas, img[6],feat_for_pred_bias=feat_for_pred_bias)['fused_bev_feat']
        if self.scene_his_fusion and self.scene_his_after_vox_his:
            start_of_sequence = torch.BoolTensor([
                    single_img_metas['start_of_sequence'] 
                    for single_img_metas in img_metas]).to(img[0][0].device)
            bev_shape=bev_feat.shape[2:]
            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],-1)

            bev_feat=self.scene_his_func_after_vox_his(bev_feat,start_of_sequence)
            
            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],*bev_shape)
        
        bev_feat = self.bev_encoder(bev_feat)
        if isinstance(bev_feat,dict):
            bev_feat=bev_feat['x']
            bev_feat=bev_feat[0]
       
        # load lidar height
        lidar_height_map_in  = kwargs.pop("bev_height_map", None)
        lidar_height_mask_in  = kwargs.pop("bev_height_mask", None)
        if isinstance(lidar_height_map_in,list):
            lidar_height_map_in=lidar_height_map_in[0]
            lidar_height_mask_in=lidar_height_mask_in[0]

        if len(bev_feat.shape)==5:
            lss_bev = bev_feat.mean(-1)
        else:
            lss_bev = bev_feat


        if self.training:
            if self.gts_surroundocc:
                height_map_gt  = kwargs['gt_height_map']*0.5 - 4.5 # (B, X, Y)    
            else: 
                height_map_gt  = kwargs['gt_height_map']*0.4 - 0.6 # (B, X, Y)    
            height_mask_gt  = kwargs['gt_height_mask'] # (B, X, Y)
            if self.train_iter < self.gt_period_end:
                # GT period                        
                height_map_cond = height_map_gt
                height_mask_cond = height_mask_gt
            else:
                if self.train_iter < self.mix_period_end:
                    # Mixing period
                    height_map_mixed = lidar_height_map_in.clone()
                    height_mask_mixed = lidar_height_mask_in.clone()

                    if self.curr_mode == 'replace':
                        rand_replace_mask = torch.rand(
                            height_mask_gt.shape,
                            device=height_mask_gt.device
                        )
                        replace_mask = (rand_replace_mask < self.height_denoise_rate) & height_mask_gt
                        height_map_mixed[replace_mask] = height_map_gt[replace_mask]
                        height_mask_mixed[replace_mask] = True

                        height_map_cond = height_map_mixed
                        height_mask_cond = height_mask_mixed
                    
                    elif self.curr_mode == 'linear_interpolation':
                        alpha = float(self.height_denoise_rate)

                        height_map_cond = lidar_height_map_in + alpha * (
                            height_map_gt - lidar_height_map_in
                        )

                        height_map_cond = torch.where(
                            lidar_height_mask_in,
                            height_map_cond,
                            lidar_height_map_in
                        )
                        height_mask_cond = lidar_height_mask_in

                    else:
                        raise ValueError("Wrong curr_mode !!!")

                else:
                    # Pred period
                    height_map_cond = lidar_height_map_in
                    height_mask_cond = lidar_height_mask_in

        # NOTE: testing only use lidar
        else:
            height_map_cond = lidar_height_map_in
            height_mask_cond = lidar_height_mask_in

        if self.with_specific_component('backward_projection') and self.with_specific_component('fuse'):
            if self.geometry_group:
                geometry=geometry.reshape(geometry.shape[0],geometry.shape[1]//self.geometry_group,self.geometry_group,*geometry.shape[2:])
                geometry=geometry.sum(2)/self.geometry_group
            bev_feat_refined = self.backward_projection(
                                        mlvl_feats=[context],
                                        pred_img_depth=geometry,
                                        lss_bev=lss_bev,  # occ_2d_out_channels
                                        cam_params=cam_params,
                                        img_metas=img_metas,
                                        bev_height_map=height_map_cond,
                                        bev_height_mask=height_mask_cond)

            bev_feat = self.fuse(bev_feat, bev_feat_refined) 

        if self.occ_backbone_2d :
            if self.with_cp and bev_feat[0].requires_grad:
                bev_feat = checkpoint(self.final_conv, bev_feat)
            else:
                bev_feat = self.final_conv(bev_feat)

            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1]//self.dz,self.dz,*bev_feat.shape[2:])
            bev_feat=bev_feat.permute(0,1,3,4,2)
            bev_feat=[bev_feat]
        else:
            # NOTE:
            bev_feat=[bev_feat]

        return_map['img_bev_feat'] = bev_feat
       
        return return_map





