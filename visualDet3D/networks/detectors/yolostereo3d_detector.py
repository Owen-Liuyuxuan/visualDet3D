import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from visualDet3D.networks.utils.registry import DETECTOR_DICT
from visualDet3D.utils.timer import profile
from visualDet3D.networks.heads import losses
from visualDet3D.networks.detectors.yolostereo3d_core import YoloStereo3DCore
from visualDet3D.networks.heads.detection_3d_head import StereoHead
from visualDet3D.networks.lib.blocks import AnchorFlatten, ConvBnReLU
from visualDet3D.networks.backbones.resnet import BasicBlock



@DETECTOR_DICT.register_module
class Stereo3D(nn.Module):
    """
        Stereo3D
    """
    def __init__(self, network_cfg):
        super(Stereo3D, self).__init__()

        self.obj_types = network_cfg.obj_types

        self.build_head(network_cfg)

        self.build_core(network_cfg)

        self.network_cfg = network_cfg

    def build_core(self, network_cfg):
        self.core = YoloStereo3DCore(network_cfg.backbone)

    def build_head(self, network_cfg):
        self.bbox_head = StereoHead(
            **(network_cfg.head)
        )

        self.disparity_loss = losses.DisparityLoss(maxdisp=96)

    def train_forward(self, left_images, right_images, annotations, P2, P3, disparity=None):
        """
        Args:
            img_batch: [B, C, H, W] tensor
            annotations: check visualDet3D.utils.utils compound_annotation
            calib: visualDet3D.kitti.data.kitti.KittiCalib or anything with obj.P2
        Returns:
            cls_loss, reg_loss: tensor of losses
            loss_dict: [key, value] pair for logging
        """
        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        cls_loss, reg_loss, loss_dict = self.bbox_head.loss(cls_preds, reg_preds, anchors, annotations, P2)

        if reg_loss.mean() > 0 and not disparity is None and not depth_output is None:
            disp_loss = 1.0 * self.disparity_loss(depth_output, disparity)
            loss_dict['disparity_loss'] = disp_loss
            reg_loss += disp_loss

            self.depth_output = depth_output.detach()
        else:
            loss_dict['disparity_loss'] = torch.zeros_like(reg_loss)
        return cls_loss, reg_loss, loss_dict

    def test_forward(self, left_images, right_images, P2, P3):
        assert left_images.shape[0] == 1 # we recommmend image batch size = 1 for testing

        output_dict = self.core(torch.cat([left_images, right_images], dim=1))
        depth_output   = output_dict['depth_output']

        cls_preds, reg_preds = self.bbox_head(
                dict(
                    features=output_dict['features'],
                    P2=P2,
                    image=left_images
                )
            )

        anchors = self.bbox_head.get_anchor(left_images, P2)

        scores, bboxes, cls_indexes = self.bbox_head.get_bboxes(cls_preds, reg_preds, anchors, P2, left_images)
        
        return scores, bboxes, cls_indexes


    def forward(self, inputs):

        if isinstance(inputs, list) and len(inputs) >= 5:
            return self.train_forward(*inputs)
        else:
            return self.test_forward(*inputs)
