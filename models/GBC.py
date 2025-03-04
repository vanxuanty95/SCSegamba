'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from torch import nn as nn

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

class GBC(nn.Module):
    def __init__(self, in_channels, norm_type = 'GN') -> None:
        super().__init__()

        self.proj = BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=in_channels//16)
        self.nonliner = nn.ReLU()

        self.proj2 = BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm2 = nn.GroupNorm(num_channels=in_channels, num_groups=in_channels//16)
        self.nonliner2 = nn.ReLU()

        self.proj3 = BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm3 = nn.GroupNorm(num_channels=in_channels, num_groups=in_channels//16)
        self.nonliner3 = nn.ReLU()

        self.proj4 = BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        if(norm_type == 'GN'):
            self.norm4 = nn.GroupNorm(num_channels=in_channels, num_groups=16)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1_1 = self.proj(x)
        x1_1 = self.norm(x1_1)
        x1_1 = self.nonliner(x1_1)

        x1 = self.proj2(x1_1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 * x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        return x + x_residual

