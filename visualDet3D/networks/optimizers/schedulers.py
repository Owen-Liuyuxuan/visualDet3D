from easydict import EasyDict 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Union

class PolyLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, gamma=0.9, n_iteration=-1):
        self.step_size = n_iteration
        self.gamma = gamma
        super(PolyLR, self).__init__(optimizer)

    def get_lr(self):
        decay = 1 - self._step_count / float(self.step_size)
        decay = max(0., decay) ** self.gamma
        return [base_lr * decay for base_lr in self.base_lrs]


class GradualWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    
    From:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """

    def __init__(self, optimizer, multiplier:float, total_epoch:int, after_scheduler_cfg=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = build_scheduler(after_scheduler_cfg, optimizer)
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != optim.lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def build_scheduler(scheduler_cfg:Union[EasyDict, None], optimizer)->optim.lr_scheduler._LRScheduler:
    if scheduler_cfg is None:
        return optim.lr_scheduler.ExponentialLR(optimizer, 1.0)
    if scheduler_cfg.type_name.lower() == 'StepLR'.lower():
        return optim.lr_scheduler.StepLR(optimizer, **scheduler_cfg.keywords)
    if scheduler_cfg.type_name.lower() == 'MultiStepLR'.lower():
        return optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_cfg.keywords)
    if scheduler_cfg.type_name.lower() == 'ExponentialLR'.lower():
        return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_cfg.keywords)
    if scheduler_cfg.type_name.lower() == 'CosineAnnealingLR'.lower():
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_cfg.keywords)
    if scheduler_cfg.type_name.lower() == 'PolyLR'.lower():
        return PolyLR(optimizer, **scheduler_cfg.keywords)
    if scheduler_cfg.type_name.lower() == 'GradualWarmupScheduler'.lower():
        return GradualWarmupScheduler(optimizer, **scheduler_cfg.keywords)
    
    raise NotImplementedError(scheduler_cfg)
