import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
def build_tensor_grid(shape):
    """
        input:
            shape = (h, w)
        output:
            yy_grid = (h, w)
            xx_grid = (h, w)
    """
    h, w = shape[0], shape[1]
    x_range = np.arange(h, dtype=np.float32)
    y_range = np.arange(w, dtype=np.float32)
    yy, xx  = np.meshgrid(y_range, x_range)
    yy_grid = 2.0 * yy / float(h) - 1
    xx_grid = 2.0 * xx / float(w) - 1
    return yy_grid, xx_grid

class CoordinateConv(nn.Module):
    """
        CoordinateConv:
            https://arxiv.org/pdf/1807.03247.pdf    
        build with our own understanding
    """
    def __init__(self, num_feature_in, num_feature_out, kernel_size=3, dilation=1, stride=1, padding=None):
        super(CoordinateConv, self).__init__()
        if padding is None:
            padding = dilation * int((kernel_size - 1) / 2)
        self.main_conv = nn.Conv2d(num_feature_in+2, num_feature_out,
                                   kernel_size, stride=stride, dilation=dilation,
                                   padding=0)
        self.padding = nn.ZeroPad2d(padding)
        self.norm    = nn.BatchNorm2d(num_feature_out)
        #self.coord_conv = nn.Conv2d(2, num_feature_out, kernel_size, stride=stride, dilation=dilation,
        #                            padding=padding, bias=False)
        #self.coord_conv.weight.data.fill_(0.0)
        self.shape = None

    def forward(self, x):
        if self.shape is None or not self.shape == x.shape:
            self.shape = x.shape
            self.grid = np.stack(build_tensor_grid([x.shape[2], x.shape[3]]), axis=0) #[2, h, w]
            self.grid = x.new(self.grid).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) #[1, 2, h, w]
        x = torch.cat([x, self.grid], dim=1)
        x = self.padding(x)
        x = self.main_conv(x)
        x = self.norm(x)
        #x += self.coord_conv(grid)
        return x
class ResCoordinateConv(nn.Module):
    def __init__(self, num_feature_in, num_feature_out, kernel_size=3, dilation=1, stride=1, padding=None):
        super(ResCoordinateConv, self).__init__()
        if padding is None:
            padding = dilation * int((kernel_size - 1) / 2)
        self.base_conv = nn.Sequential(
            nn.Conv2d(num_feature_in, num_feature_out, kernel_size, dilation=dilation, stride=stride, padding=padding),
            nn.BatchNorm2d(num_feature_out),
            nn.ReLU(),
        )
        self.coord_conv = CoordinateConv(num_feature_out, num_feature_out, kernel_size, dilation, 1, padding)

    def forward(self, x):
        x = self.base_conv(x)
        x1 = self.coord_conv(x)
        return F.relu(x+x1)

class DisparityConv(nn.Module):
    """ Disparity version of CoordinateConv.

    Different from coordinateConv, DisparityConv produce extra disparity features instead of normalized coordinate. 

    The disparity is computed as:

    Assume a pixel in the picture is $relative_evelation meters below the camera center,
    project the pixel back to world and compute the distance between the camera and the projected point;
    Assume a virtual stereo baseline is $baseline meters, compute the disparity of that point.
    Encode the disparity value in a one-hot style.
    
    If there is no solution, disparity value will simply be zero.
    """
    def __init__(self, num_feature_in, num_feature_out, kernel_size=3, dilation=1, stride=1, padding=None, relative_elevation=1.65, baseline=0.54, max_disp=192, relu=True):
        super(DisparityConv, self).__init__()
        if padding is None:
            padding = dilation * int((kernel_size - 1) / 2)
        self.relative_elevation = relative_elevation
        self.baseline = baseline
        self.max_disp = max_disp
        self.model = nn.Sequential(
            nn.Conv2d(num_feature_in + 1, num_feature_out, kernel_size, dilation=dilation, stride=stride, padding=padding),
            nn.BatchNorm2d(num_feature_out),
            nn.ReLU() if relu else nn.Sequential()
        )

    def forward(self, x:torch.Tensor, P2:torch.Tensor):
        """ Forward method
        Args:
            x: input features [B, C, H, W], torch.tensor
            P2: torch.tensor, calibration matrix, [B, 3, 4]
        """
        h, w = x.shape[2], x.shape[3]
        x_range = np.arange(h, dtype=np.float32)
        y_range = np.arange(w, dtype=np.float32)
        _, yy_grid  = np.meshgrid(y_range, x_range)

        yy_grid =  x.new(yy_grid).unsqueeze(0) #[1, H, W]
        fy =  P2[:, 1:2, 1:2] #[B, 1, 1]
        cy =  P2[:, 1:2, 2:3] #[B, 1, 1]
        Ty =  P2[:, 1:2, 3:4] #[B, 1, 1]
        
        disparity = fy * self.baseline  * (yy_grid - cy) / (torch.abs(fy * self.relative_elevation + Ty) + 1e-10)
        disparity = F.relu(disparity)
        # disparity = torch.where(
        #     disparity > self.max_disp,
        #     torch.ones_like(disparity) * (self.max_disp - 1),
        #     disparity
        # ).long() #[B, H, W]

        # disparity = F.one_hot(disparity, self.max_disp).float().permute(0, 3, 1, 2).contiguous()
        x = torch.cat([x, disparity.unsqueeze(1)], dim=1)
        x = self.model(x)

        return x

if __name__ == "__main__":
    layer = ResCoordinateConv(256, 256).cuda()
    A = torch.rand(2, 256, 18, 40).cuda()
    B = layer(A)
    assert(B.shape == A.shape)

    layer2 = DisparityConv(256, 256, max_disp=192 // 16).cuda()
    A = torch.rand(2, 256, 18, 80).cuda()
    P2_1 = torch.tensor(
        [[20, 0, 40, 0],
         [0 ,20, 5, 0],
         [0, 0, 1, 0]
        ], dtype=torch.float32
    ).cuda()
    P2 = torch.stack([P2_1, P2_1.clone()], dim=0)
    B = layer2(A, P2)
    assert(B.shape == A.shape)
