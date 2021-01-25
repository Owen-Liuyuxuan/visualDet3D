import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import resnet


class YoloMono3DCore(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCore, self).__init__()
        self.backbone =resnet(**backbone_arguments)
        
    def forward(self, x):
        x = self.backbone(x['image'])
        x = x[0]
        return x
