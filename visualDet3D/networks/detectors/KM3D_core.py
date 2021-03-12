import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import build_backbone

class KM3DCore(nn.Module):
    """Some Information about RTM3D_core"""
    def __init__(self, backbone_arguments=dict()):
        super(KM3DCore, self).__init__()
        self.backbone = build_backbone(backbone_arguments)
        output_features = 2048 if backbone_arguments['depth'] > 34 else 512
        feature_size = 256

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
        x = self.deconv_layers(x[0])
        return x
