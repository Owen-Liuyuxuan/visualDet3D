import numpy as np
import torch.nn as nn
import torch
import math
import time
from torchvision.ops import nms
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.utils.utils import BBoxTransform, ClipBoxes
from visualDet3D.networks.heads.anchors import Anchors
from visualDet3D.networks.heads.retinanet_head import RetinanetHead
from visualDet3D.networks.heads import losses
from visualDet3D.networks.lib.blocks import ConvBnReLU
from visualDet3D.networks.backbones import resnet

class FPN(nn.Module):
    """Some Information about FPN"""
    def __init__(self, in_channels, out_channels, num_outs):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels[i], out_channels, 1) for i in range(len(in_channels))
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, (3, 3), padding=1) for i in range(len(in_channels))
            ]
        )

        extra_levels = num_outs - len(in_channels) # add extra down-sampling, Retinanet add convs on inputs
        if extra_levels > 0:
            for i in range(extra_levels):
                if i == 0:
                    self.fpn_convs.append(
                        nn.Conv2d(in_channels[-1], out_channels, 3, padding=1, stride=2)
                    )
                else:
                    self.fpn_convs.append(
                        nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=2)
                    )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)

        # Build Laterals
        laterals = [
            self.lateral_convs[i](feats[i]) for i in range(len(self.in_channels))
        ]

        # top-down path
        for i in range(len(self.in_channels) - 1, 0, -1):
            laterals[i - 1] += torch.nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )
        # build original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(self.in_channels))
        ]
        if len(self.fpn_convs) > len(outs):
            # RetinaNet add convolutions to inputs
            outs.append(self.fpn_convs[len(outs)](feats[-1]))

            for i in range(len(outs), len(self.fpn_convs)):
                outs.append(self.fpn_convs[i](outs[-1])) # default no relu in mmdetection retinanet, with relu in pytorch/retinanet

        return tuple(outs)

class RetinaNetCore(nn.Module):
    """Some Information about RetinaNetCore"""
    def __init__(self, backbone_cfg, neck_cfg):
        super(RetinaNetCore, self).__init__()
        self.backbone = resnet(**backbone_cfg)
        self.neck     = FPN(**neck_cfg)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return feats

@DETECTOR_DICT.register_module
class RetinaNet(nn.Module):
    """
        RetinaNet for 2D object detection. Learning from mmdetection but fit in our resign.
    """
    def __init__(self, network_cfg):
        super(RetinaNet, self).__init__()

        self.clipBoxes = ClipBoxes()

        self.obj_types = network_cfg.obj_types

        self.build_core(network_cfg)

        self.build_head(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = RetinaNetCore(network_cfg.backbone, network_cfg.neck)

    def build_head(self, network_cfg):
        self.bbox_head = RetinanetHead(**(network_cfg.head) )


    def training_forward(self, img_batch, annotations):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """

        feats = self.core(img_batch)

        cls_preds, reg_preds = self.bbox_head(feats)
        anchors = self.bbox_head.get_anchor(img_batch)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations)
        return (cls_loss, reg_loss, loss_dict)
    
    def test_forward(self, img_batch):
        """
        Args:
            img_batch: [B, C, H, W] tensor
        Returns:
            results: a nested list:
                result[i] = detection_results for obj_types[i]
                    each detection result is a list [scores, bbox, obj_type]:
                        bbox = [bbox2d(length=4) , cx, cy, z, w, h, l, alpha]
        """
        assert img_batch.shape[0] == 1 # we recommmend image batch size = 1 for testing

        feats = self.core(img_batch)
        cls_preds, reg_preds = self.bbox_head(feats)
        anchors = self.bbox_head.get_anchor(img_batch)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch)