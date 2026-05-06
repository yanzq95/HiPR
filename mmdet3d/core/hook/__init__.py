# Copyright (c) OpenMMLab. All rights reserved.
from .ema import MEGVIIEMAHook
from .utils import is_parallel
from .sequentialsontrol import SequentialControlHook
from .sequentialsontrol_flow import SequentialControlHookFlow
from .syncbncontrol import SyncbnControlHook
from .fusionweightcontrol import FusionRateControlHook
from .fusionweightcontrol_depth import FusionRateControlDepthHook
from .fusionweightcontrol_pose import FusionRateControlPoseHook
# HiPR
from .fusionweightcontrol_height import FusionRateControlHeightHook
from .trainingiter import GetTrainingIter

__all__ = ['MEGVIIEMAHook', 'is_parallel', 'SequentialControlHook',  'SequentialControlHookFlow',
           'SyncbnControlHook','FusionRateControlHook','FusionRateControlDepthHook','FusionRateControlPoseHook']
