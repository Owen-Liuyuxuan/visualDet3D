import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.utils import DETECTOR_DICT
from visualDet3D.networks.detectors.yolomono3d_core import YoloMono3DCore
from visualDet3D.networks.heads.detection_3d_head import AnchorBasedDetection3DHead
from visualDet3D.networks.lib.blocks import AnchorFlatten
from visualDet3D.networks.lib.look_ground import LookGround

class GroundAwareHead(AnchorBasedDetection3DHead):
    def init_layers(self, num_features_in,
                          num_anchors:int,
                          num_cls_output:int,
                          num_reg_output:int,
                          cls_feature_size:int=1024,
                          reg_feature_size:int=1024,
                          **kwargs):
        self.cls_feature_extraction = nn.Sequential(
            nn.Conv2d(num_features_in, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(cls_feature_size, cls_feature_size, kernel_size=3, padding=1),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),

            nn.Conv2d(cls_feature_size, num_anchors*(num_cls_output), kernel_size=3, padding=1),
            AnchorFlatten(num_cls_output)
        )
        self.cls_feature_extraction[-2].weight.data.fill_(0)
        self.cls_feature_extraction[-2].bias.data.fill_(0)

        self.reg_feature_extraction = nn.Sequential(
            LookGround(reg_feature_size),
            nn.Conv2d(num_features_in, reg_feature_size, 3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(),
            nn.Conv2d(reg_feature_size, reg_feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(reg_feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_feature_size, num_anchors*num_reg_output, kernel_size=3, padding=1),
            AnchorFlatten(num_reg_output)
        )

        self.reg_feature_extraction[-2].weight.data.fill_(0)
        self.reg_feature_extraction[-2].bias.data.fill_(0)

    def forward(self, inputs):
        cls_preds = self.cls_feature_extraction(inputs['features'])
        reg_preds = self.reg_feature_extraction(inputs)

        return cls_preds, reg_preds

@DETECTOR_DICT.register_module
class Yolo3D(nn.Module):
    """
        YoloMono3DNetwork
    """
    def __init__(self, network_cfg):
        super(Yolo3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg


    def build_core(self, network_cfg):
        self.core = YoloMono3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = AnchorBasedDetection3DHead(
            **(network_cfg.head)
        )

    def training_forward(self, img_batch, annotations, P2):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """

        features  = self.core(dict(image=img_batch, P2=P2))
        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2, image=img_batch))

        anchors = self.bbox_head.get_anchor(img_batch, P2)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        return cls_loss, reg_loss, loss_dict
    
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
        cls_preds, reg_preds = self.bbox_head(dict(features=features, P2=P2))

        anchors = self.bbox_head.get_anchor(img_batch, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, img_batch)

        return scores, bboxes, cls_indexes

    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) == 3:
            img_batch, annotations, calib = inputs
            return self.training_forward(img_batch, annotations, calib)
        else:
            img_batch, calib = inputs
            return self.test_forward(img_batch, calib)

@DETECTOR_DICT.register_module
class GroundAwareYolo3D(Yolo3D):
    """Some Information about GroundAwareYolo3D"""

    def build_head(self, network_cfg):
        self.bbox_head = GroundAwareHead(
            **(network_cfg.head)
        )

