import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
from torchvision.ops import nms
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.detectors.KM3D_core import KM3DCore
from visualDet3D.networks.heads.km3d_head import KM3DHead
from visualDet3D.networks.heads.monoflex_head import MonoFlexHead
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.look_ground import LookGround
from visualDet3D.networks.lib.ops.dcn.deform_conv import DeformConv

@DETECTOR_DICT.register_module
class KM3D(nn.Module):
    """
        KM3D
    """
    def __init__(self, network_cfg):
        super(KM3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg


    def build_core(self, network_cfg):
        self.core = KM3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = KM3DHead(
            **(network_cfg.head)
        )

    def training_forward(self, img_batch, annotations, meta):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            meta:
                calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
                epoch: current_epoch
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """

        features  = self.core(dict(image=img_batch, P2=meta['P2']))
        output_dict = self.bbox_head(features)

        loss, loss_dict = self.bbox_head.loss(output_dict, annotations, meta)

        return loss, loss_dict
    
    def test_forward(self, img_batch, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        features  = self.core(dict(image=img_batch, P2=P2))
        output_dict = self.bbox_head(features)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(output_dict, P2, img_batch)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, meta = inputs
            return self.training_forward(img_batch, annotations, meta)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)

@DETECTOR_DICT.register_module
class MonoFlex(KM3D):
    """
        MonoFlex
    """
    def build_head(self, network_cfg):
        self.bbox_head = MonoFlexHead(**(network_cfg.head))

