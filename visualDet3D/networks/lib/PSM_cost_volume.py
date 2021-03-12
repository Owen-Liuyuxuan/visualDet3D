"""
    This script implements cost_volume module in the PSM networks
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from visualDet3D.utils.timer import profile
def make_grid(grid_shape):
    
    #grid: (y, x, z)
    grid_1ds = [torch.arange(-1, 1, 2.0/shape) for shape in grid_shape]
    grids = torch.meshgrid(grid_1ds)
    return grids

class CostVolume(nn.Module):
    """
        While PSV module define depth dimension similar to the depth in real world

        Cost Volume implementation in PSM network and its prior networks define this directly as disparity
    """
    def __init__(self, max_disp=192, downsample_scale=4, input_features=1024, PSM_features=64):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_features, PSM_features, 1),
            nn.BatchNorm2d(PSM_features),
            nn.ReLU(),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(2 * PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
            nn.Conv3d(PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
        )
        self.output_channel = PSM_features * self.depth_channel
    @profile("Cost Volume", 1, 10)
    def forward(self, left_features, right_features):
        batch_size, _, w, h = left_features.shape
        left_features = self.down_sample(left_features)
        right_features = self.down_sample(right_features)
        cost = Variable(
            torch.FloatTensor(left_features.size()[0],
                              left_features.size()[1]*2,
                              self.depth_channel,
                              left_features.size()[2],  
                              left_features.size()[3]).zero_(), 
            volatile= not self.training
        ).cuda()

        for i in range(self.depth_channel):
            if i > 0 :
                 cost[:, :left_features.size()[1], i, :,i:]  = left_features[:,:,:,i:]
                 cost[:, left_features.size()[1]:, i, :,i:]  = right_features[:,:,:,:-i]
            else:
                 cost[:, :left_features.size()[1], i, :,:]   = left_features
                 cost[:, left_features.size()[1]:, i, :,:]   = right_features
        cost = cost.contiguous()
        cost = self.conv3d(cost) # .squeeze(1)
        cost = cost.reshape(batch_size, -1, w, h).contiguous()
        return cost


class PSMCosineModule(nn.Module):
    """Some Information about PSMCosineModule"""
    def __init__(self, max_disp=192, downsample_scale=4, input_features=512):
        super(PSMCosineModule, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        #self.distance_function = nn.CosineSimilarity(dim=1)

    @profile("PSM Cos Volume", 1, 20)
    def forward(self, left_features, right_features):
        cost = Variable(
            torch.FloatTensor(left_features.size()[0],
                              self.depth_channel,
                              left_features.size()[2],  
                              left_features.size()[3]).zero_(), 
            volatile= not self.training
        ).cuda()

        for i in range(self.depth_channel):
            if i > 0 :
                 cost[:, i, :,i:]  = (left_features[:,:,:,i:] * right_features[:,:,:,:-i]).mean(dim=1)
            else:
                 cost[:, i, :, :]  = (left_features * right_features).mean(dim=1)
        cost = cost.contiguous()
        return cost

class DoublePSMCosineModule(PSMCosineModule):
    """Some Information about DoublePSMCosineModule"""
    def __init__(self, max_disp=192, downsample_scale=4):
        super(DoublePSMCosineModule, self).__init__(max_disp=max_disp, downsample_scale=downsample_scale)
        self.depth_channel = self.depth_channel

    def forward(self, left_features, right_features):
        b, c, h, w = left_features.shape
        base_grid_y, base_grid_x = make_grid(right_features.shape[2:]) #[h, w]
        base_grid_x = base_grid_x - 1.0 / right_features.shape[1] 
        shifted_grid = torch.stack([base_grid_y, base_grid_x], dim=-1).cuda().unsqueeze(0).repeat(b, 1, 1, 1)
        right_features_shifted = F.grid_sample(right_features, shifted_grid)
        cost_1 = super(DoublePSMCosineModule, self)(left_features, right_features)
        cost_2 = super(DoublePSMCosineModule, self)(left_features, right_features_shifted)
        return torch.cat([cost_1, cost_2], dim=1)



if __name__ == "__main__":
    model = DoublePSMCosineModule(max_disp=192, downsample_scale=16).cuda()
    left_feature = torch.randn(2, 128, 12, 56, requires_grad=True, device="cuda:0")
    right_feature = torch.randn(2, 128, 12, 56, requires_grad=True, device="cuda:0")
    output = model(left_feature, right_feature, 0, 0) #currently dummy
    mean_1 = output.mean() 
    mean_1.backward()
    print(left_feature.grad.std())
    print(model.depth_channel)
    print(output.shape)
    import time
    start = time.time()
    for _ in range(10):
        output = model(left_feature, right_feature, 0, 0)
    print(time.time() - start)
