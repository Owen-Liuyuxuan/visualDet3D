import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import build_backbone
from visualDet3D.networks.backbones.dla import DLA
from visualDet3D.networks.backbones.dla_utils import DLASegUpsample

class KM3DCore(nn.Module):
    """Some Information about RTM3D_core"""
    def __init__(self, backbone_arguments=dict()):
        super(KM3DCore, self).__init__()
        self.backbone = build_backbone(backbone_arguments)
        
        if backbone_arguments['name'].lower() == 'vit':
            output_features = backbone_arguments['dim']
        if backbone_arguments['name'].lower() == 'resnet' or backbone_arguments['name'].lower() == 'dla':
            if backbone_arguments['depth'] > 34:
                output_features = 2024
            elif backbone_arguments['depth'] <= 34:
                output_features = 512
        
        if isinstance(self.backbone, DLA):
            feature_size = 64
            print(f"Apply DLA Upsampling instead, feature_size={feature_size}")
            self.deconv_layers = DLASegUpsample(
                input_channels=[16, 32, 64, 128, 256, 512],
                down_ratio=4,
                final_kernel=1,
                last_level=5,
                out_channel=64, ## Notice that if in DLA the head_feature_size should be 256 and input features should be 64 for the heads.
            )
        else:
            feature_size = 256
            print(f"Apply Baseline ConvTranspose Upsampling instead, feature_size={feature_size}")
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(output_features, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(feature_size, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(feature_size, feature_size, (4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(feature_size),
                nn.ReLU(inplace=True),
            )
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                
    def forward(self, x):
        x = self.backbone(x['image'])
        if isinstance(self.backbone, DLA):
            x = self.deconv_layers(x)
        else:
            x = self.deconv_layers(x[-1])
        return x
