from mmcv.runner import BaseModule
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.models.builder import NECKS
import torch
import torch.nn as nn
import torch.nn.functional as F

@NECKS.register_module ()
class Add (BaseModule):
    def __init__(self):
        super (Add, self).__init__ ()

    def forward(self, forward_feat, backward_feat):
        """
        forward_feat: (B, C, X, Y, Z) LSS project with height
        backward_feat: (B, C, X, Y) BEVFomer without height
        """


        if len(forward_feat.shape) == 4:
            return forward_feat + backward_feat
        else:
            if len(forward_feat.shape) ==  5:
                return forward_feat + backward_feat[...,None]
            else:
                return ValueError("wrong dim in forward and backward fuse")