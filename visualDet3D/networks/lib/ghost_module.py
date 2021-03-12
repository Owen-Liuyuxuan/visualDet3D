"""
    This script implement ghost module from 
    "GhostNet: More Features from Cheap Operations"
    https://arxiv.org/pdf/1911.11907.pdf
    Introduction in:
    https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/Building_Blocks/GhostNet/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class GhostModule(nn.Module):
    """
        Ghost Module from https://github.com/iamhankai/ghostnet.pytorch.

    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Sequential(),
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class ResGhostModule(GhostModule):
    """Some Information about ResGhostModule"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        assert(ratio > 2)
        super(ResGhostModule, self).__init__(inp, oup-inp, kernel_size, ratio-1, dw_size, relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsampling = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsampling = None

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if not self.downsampling is None:
            x = self.downsampling(x)
        out = torch.cat([x, x1, x2], dim=1)
        return out[:,:self.oup,:,:]
