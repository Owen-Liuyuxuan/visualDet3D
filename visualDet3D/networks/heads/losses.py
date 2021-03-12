from easydict import EasyDict
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from visualDet3D.networks.utils.utils import calc_iou
from visualDet3D.networks.lib.disparity_loss import stereo_focal_loss
from visualDet3D.utils.timer import profile


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, balance_weights=torch.tensor([1.0], dtype=torch.float)):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.register_buffer("balance_weights", balance_weights)

    def forward(self, classification:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:Optional[float]=None, 
                      balance_weights:Optional[torch.Tensor]=None)->torch.Tensor:
        """
            input:
                classification  :[..., num_classes]  linear output
                targets         :[..., num_classes] == -1(ignored), 0, 1
            return:
                cls_loss        :[..., num_classes]  loss with 0 in trimmed or ignored indexes 
        """
        if gamma is None:
            gamma = self.gamma
        if balance_weights is None:
            balance_weights = self.balance_weights

        probs = torch.sigmoid(classification) #[B, N, 1]
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        bce = -(targets * nn.functional.logsigmoid(classification)) * balance_weights - ((1-targets) * nn.functional.logsigmoid(-classification)) #[B, N, 1]
        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6

        return cls_loss

class SoftmaxFocalLoss(nn.Module):
    def forward(self, classification:torch.Tensor, 
                      targets:torch.Tensor, 
                      gamma:float, 
                      balance_weights:torch.Tensor)->torch.Tensor:
        ## Calculate focal loss weights
        probs = torch.softmax(classification, dim=-1)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
        focal_weight = torch.pow(focal_weight, gamma)

        bce = -(targets * torch.log_softmax(classification, dim=1))

        cls_loss = focal_weight * bce

        ## neglect  0.3 < iou < 0.4 anchors
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

        ## clamp over confidence and correct ones to prevent overfitting
        cls_loss = torch.where(torch.lt(cls_loss, 1e-5), torch.zeros(cls_loss.shape).cuda(), cls_loss) #0.02**2 * log(0.98) = 8e-6
        cls_loss = cls_loss * balance_weights
        return cls_loss


class ModifiedSmoothL1Loss(nn.Module):
    def __init__(self, L1_regression_alpha:float):
        super(ModifiedSmoothL1Loss, self).__init__()
        self.alpha = L1_regression_alpha

    def forward(self, normed_targets:torch.Tensor, pos_reg:torch.Tensor):
        regression_diff = torch.abs(normed_targets - pos_reg) #[K, 12]
        ## Smoothed-L1 formula:
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / self.alpha),
            0.5 * self.alpha * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.alpha
        )
        ## clipped to avoid overfitting
        regression_loss = torch.where(
           torch.le(regression_diff, 0.01),
           torch.zeros_like(regression_loss),
           regression_loss
        )

        return regression_loss

class IoULoss(nn.Module):
    """Some Information about IoULoss"""
    def forward(self, preds:torch.Tensor, targets:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        """IoU Loss

        Args:
            preds (torch.Tensor): [x1, y1, x2, y2] predictions [*, 4]
            targets (torch.Tensor): [x1, y1, x2, y2] targets [*, 4]

        Returns:
            torch.Tensor: [-log(iou)] [*]
        """
        
        # overlap
        lt = torch.max(preds[..., :2], targets[..., :2])
        rb = torch.min(preds[..., 2:], targets[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        # union
        ap = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
        ag = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union
        ious = torch.clamp(ious, min=eps)
        return -ious.log()

class DisparityLoss(nn.Module):
    """Some Information about DisparityLoss"""
    def __init__(self, maxdisp:int=64):
        super(DisparityLoss, self).__init__()
        #self.register_buffer("disp",torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])))
        self.criterion = stereo_focal_loss.StereoFocalLoss(maxdisp)

    def forward(self, x:torch.Tensor, label:torch.Tensor)->torch.Tensor:
        #x = torch.softmax(x, dim=1)
        label = label.cuda().unsqueeze(1)
        loss = self.criterion(x, label, variance=0.5)
        #mask = (label > 0) * (label < 64)
        #loss = nn.functional.smooth_l1_loss(disp[mask], label[mask])
        return loss