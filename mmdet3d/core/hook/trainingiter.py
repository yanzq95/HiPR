from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
import math
__all__ = ['GetTrainingIter']


@HOOKS.register_module()
class GetTrainingIter(Hook):
    """ """

    def __init__(self,):
        super().__init__()

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.train_iter=flag
        else:
            runner.model.module.train_iter=flag

    def before_train_iter(self, runner):
        curr_step = runner.iter
        self.set_temporal_flag(runner,curr_step)


