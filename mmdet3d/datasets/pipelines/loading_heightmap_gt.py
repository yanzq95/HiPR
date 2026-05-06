# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Dubing Chen
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import os.path as osp
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from copy import deepcopy
import cv2
import os
from torchvision.transforms.functional import rotate
from mmcv.parallel.data_container import DataContainer
import torch.nn.functional as F
import torchvision.transforms.functional as TF
occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm
def get_scene_token( info):
    if 'scene_token' in info:
        scene_name = info['scene_token']
    else:
        scene_name = info['occ_path'].split('openocc_v2/')[-1].split('/')[0]
    return scene_name
def get_ego_from_lidar(info):
    ego_from_lidar = trans_matrix(
        np.array(info['lidar2ego_translation']), 
        Quaternion(info['lidar2ego_rotation']))

    return ego_from_lidar

def get_global_pose( info, inverse=False):

    global_from_ego = trans_matrix(
        np.array(info['ego2global_translation']), 
        Quaternion(info['ego2global_rotation']))

    ego_from_lidar = trans_matrix(
        np.array(info['lidar2ego_translation']), 
        Quaternion(info['lidar2ego_rotation']))

    pose = global_from_ego.dot(ego_from_lidar)
    if inverse:
        pose = np.linalg.inv(pose)
    return pose
def get_origin( info,scene_frame):

    ref_lidar_from_global =get_global_pose(info, inverse=True)
    ref_ego_from_lidar = get_ego_from_lidar(info)


    # NOTE: getting output frames
    output_origin_list = []
    for curr_index in range(len(scene_frame)):

        global_from_curr = get_global_pose(scene_frame[curr_index], inverse=False)
        ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
        origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

        origin_tf_pad = np.ones([4])
        origin_tf_pad[:3] = origin_tf  # pad to [4]
        origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T  # [3]

        # origin
        if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
            output_origin_list.append(origin_tf)

    # select 8 origins
    if len(output_origin_list) > 8:
        select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
        output_origin_list = [output_origin_list[i] for i in select_idx]


    output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]

    return DataContainer(output_origin_tensor)




@PIPELINES.register_module()
class LoadOccupancy_heightgt(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __init__(self, 
                    convert2metric=True,
                    occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                    num_classes=17,
                    ignore_nonvisible=False,
                    mask='mask_camera',
                    ignore_classes=[],
                    fix_void=True,
                    flow_path=None,
                    load_flow=False,
                    label2_path='visual/aug_gts',
                    gts_surroundocc=False,
                    gts_openoccupancy=False,
                    occ_size=[200,200,16],
                    ) :
        self.occupancy_path = occupancy_path
        self.num_classes = num_classes
        self.ignore_nonvisible = ignore_nonvisible
        self.mask = mask

        self.ignore_classes=ignore_classes

        self.fix_void = fix_void
        self.flow_path=flow_path
        self.load_flow=load_flow
        self.label2_path=label2_path
        self.gts_surroundocc=gts_surroundocc
        self.gts_openoccupancy=gts_openoccupancy
        self.occ_size=occ_size

        self.convert2metric = convert2metric


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        
        if self.gts_surroundocc:
            lidar_path=results['pts_filename'] 
            occupancy_file_path=lidar_path.replace('samples/LIDAR_TOP','gts_surroundocc/samples')
            occupancy_file_path=occupancy_file_path+'.npy'
            data = np.load(occupancy_file_path)
            occ = np.zeros(self.occ_size)
            occ[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]
            occ = np.where(occ== 0, 17, occ)
            occupancy=torch.tensor(occ.copy()).long()
            visible_mask=torch.ones_like(occupancy).bool()
        elif self.gts_openoccupancy:
            scene_token = 'scene_'+results['curr']['scene_token']
            lidar_token = results['curr']['lidar_token']
            occupancy_file_path = osp.join(self.occupancy_path.replace('gts','nuScenes-Occupancy-v0.1'), scene_token, 'occupancy',lidar_token)+'.npy'
            data = np.load(occupancy_file_path)
            occ = np.zeros(self.occ_size)
            occ[data[:, 2], data[:, 1], data[:, 0]] = data[:, 3]
            occ = np.where(occ== 0, 17, occ)
            occupancy=torch.tensor(occ.copy()).long()
            visible_mask=torch.ones_like(occupancy).bool()
        else:
            scene_name = results['curr']['scene_name']
            sample_token = results['curr']['token']


            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            data = np.load(occupancy_file_path)
            occupancy = torch.tensor(data['semantics'])
            visible_mask = torch.tensor(data[self.mask])

        ######
        occupancy_original = occupancy.clone()
        #####
        if self.ignore_nonvisible:
            occupancy[~visible_mask.to(torch.bool)] = 255
        if self.load_flow:
            flow_file_path = osp.join(self.flow_path, scene_name, sample_token, 'labels.npz')
            data = np.load(flow_file_path)
            occ_flow = torch.tensor(data['flow'])


        # to FBOcc format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])
        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)
        #########
        
        occupancy_original = occupancy_original.permute(2, 0, 1)
        occupancy_original = torch.rot90(occupancy_original, 1, [1, 2])
        occupancy_original = torch.flip(occupancy_original, [1])
        occupancy_original = occupancy_original.permute(1, 2, 0)

        if self.load_flow:
            occ_flow = occ_flow.permute(2, 0, 1,3)
            occ_flow = torch.rot90(occ_flow, 1, [1, 2])
            occ_flow=occ_flow[...,[1,0]]
            # occ_flow[...,0]=-occ_flow[...,0]

            occ_flow = torch.flip(occ_flow, [1])
            # occ_flow[...,0]=-occ_flow[...,0]
            occ_flow = occ_flow.permute(1, 2, 0,3)
        
        if self.fix_void:
            occupancy[occupancy<255] = occupancy[occupancy<255] + 1
            ########
            occupancy_original[occupancy_original<255] = occupancy_original[occupancy_original<255] + 1
            ########

        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255

        if results['rotate_bda'] != 0:
            occupancy = occupancy.permute(2, 0, 1)
            occupancy = rotate(occupancy, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            
            ######################
            occupancy_original = occupancy_original.permute(2, 0, 1)
            occupancy_original = rotate(occupancy_original, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            ######################
            if self.load_flow:
                ######################
                occ_flow = occ_flow.permute(3,2, 0, 1)
                occ_flow = rotate(occ_flow, -results['rotate_bda'], fill=float('inf')).permute(2, 3, 1,0)
                ######################

        if results['flip_dx']:
            occupancy = torch.flip(occupancy, [1])
            ############
            occupancy_original = torch.flip(occupancy_original, [1])
            ############
            if self.load_flow:
                                ############
                occ_flow = torch.flip(occ_flow, [1])
                occ_flow[...,1]=-occ_flow[...,1]
                ############
        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [0])
            ###############
            occupancy_original = torch.flip(occupancy_original, [0])
            ###############
            if self.load_flow:
                    ###############
                occ_flow = torch.flip(occ_flow, [0])
                occ_flow[...,0]=-occ_flow[...,0]


        gt_height_map, gt_height_mask = self.voxel2height(occupancy)

        results['gt_occupancy'] = occupancy
        results['visible_mask'] = visible_mask
        results['visible_mask_bev'] = (occupancy==255).sum(-1)
        results['gt_occupancy_ori'] = occupancy_original
        results['gt_height_map'] = gt_height_map
        results['gt_height_mask'] = gt_height_mask

        if self.load_flow:
            results['gt_occ_flow'] = occ_flow
        return results


    def voxel2height(self, labels, ignore=[18, 255], set_free=255):
        """
        Args:
            labels: (H, W, Z): 255:free
        Returns:
            height_map: (H, W)，最高前景体素的 z 索引；无前景则为 255
        """
        H, W, Z = labels.shape
        device = labels.device
        # 前景掩码
        fg_mask = ~torch.isin(labels, torch.tensor(ignore, device=labels.device))

        # 每个 voxel 的 z 索引
        z_indices = torch.arange(Z, device=device).view(1, 1, Z)
        # 无前景填 -1
        z_masked = torch.where(fg_mask, z_indices, torch.full_like(z_indices, -1))
        top_z = z_masked.max(dim=-1).values   # (B,H,W)
        # 无前景置 255(ignore) 或者取最高值
        height_mask = top_z >= 0
        height_map = torch.where(height_mask, top_z, torch.full_like(top_z, set_free))

        height_map = height_map.transpose(1,0)
        height_mask = height_mask.transpose(1,0)

        if self.convert2metric:
            height_map = height_map*0.4 - 0.6

        return height_map, height_mask


