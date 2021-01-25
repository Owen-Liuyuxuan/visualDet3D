import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .coordconv import ResCoordinateConv

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class ConvBnReLU(nn.Module):
    """Some Information about ConvBnReLU"""

    def __init__(self, input_features=1, output_features=1, kernel_size=(1, 1), stride=[1, 1], padding='SAME', dilation=1, groups=1, relu=True):
        super(ConvBnReLU, self).__init__()
        pad_num = int((kernel_size[0] - 1) / 2) * \
            dilation if padding.lower() == 'same' else 0
        self.sequence = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=kernel_size,
                      stride=stride, padding=pad_num, dilation=dilation, groups=groups),
            nn.BatchNorm2d(output_features),
        )
        self.relu=True

    def forward(self, x):
        x = self.sequence(x)
        if self.relu:
            return F.relu(x)
        else:
            return x


class ConvReLU(nn.Module):
    """Some Information about ConvReLU"""

    def __init__(self, input_features=1, output_features=1, kernel_size=(1, 1), stride=[1, 1], padding='SAME'):
        super(ConvReLU, self).__init__()
        pad_num = int((kernel_size[0] - 1) / 2) if padding.lower() == 'same' else 0
        self.sequence = nn.Sequential(
            nn.Conv2d(input_features, output_features,
                      kernel_size=kernel_size, stride=stride, padding=pad_num),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


class LinearBnReLU(nn.Module):
    """Some Information about LinearBnReLU"""

    def __init__(self, input_features=1,  num_hiddens=1):
        super(LinearBnReLU, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_features, num_hiddens),
            nn.GroupNorm(16, num_hiddens),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


class LinearDropoutReLU(nn.Module):
    """Some Information about LinearDropoutReLU"""

    def __init__(self, input_features=1,  num_hiddens=1, drop=0.0):
        super(LinearDropoutReLU, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(input_features, num_hiddens),
            nn.Dropout(drop),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sequence(x)
        return x


class ModifiedSmoothedL1(nn.Module):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                     |x| - 0.5 / sigma^2,    otherwise
    '''

    def __init__(self, sigma):
        super(ModifiedSmoothedL1, self).__init__()
        self.sigma2 = sigma * sigma

    def forward(self, deltas, targets, sigma=None):
        sigma2 = self.sigma2 if sigma is None else sigma * sigma
        diffs = deltas - targets

        option1 = diffs * diffs * 0.5 * sigma2
        option2 = torch.abs(diffs) - 0.5 / sigma2
        
        condition_for_1 = (diffs < (1.0/sigma2)).float()
        smooth_l1 = option1 * condition_for_1 + option2 * (1-condition_for_1)
        return smooth_l1

class AnchorFlatten(nn.Module):
    """
        Module for anchor-based network outputs,
        Init args:
            num_output: number of output channel for each anchor.

        Forward args:
            x: torch.tensor of shape [B, num_anchors * output_channel, H, W]

        Forward return:
            x : torch.tensor of shape [B, num_anchors * H * W, output_channel]
    """
    def __init__(self, num_output_channel):
        super(AnchorFlatten, self).__init__()
        self.num_output_channel = num_output_channel

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.shape[0], -1, self.num_output_channel)
        return x
    
