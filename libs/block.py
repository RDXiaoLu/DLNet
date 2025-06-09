import torch 
from torch import nn
import math
import time
from einops import rearrange
import torch.nn.functional as F
import torchvision

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
 

class CAB(nn.Module):
    def __init__(self, nc, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(nc, nc // reduction, kernel_size=1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc // reduction, nc, kernel_size=1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8
        reduction_2 = 2

        self.cab = CAB(in_channels, reduction, bias)
        
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)

        self.conv1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)

    def forward(self, y):
        y_d = self.conv1x1_1(y)
        y1 = self.conv1(y_d)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y_d

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1+y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1+y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1+y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(self.conv1x1_2(y8_1))
        y9_1 = y + y9

        return y9_1

class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        reduction_2 = 2
        layers = []
        layers.append(nn.Conv2d(in_channels// reduction_2, out_channels// reduction_2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels// reduction_2, out_channels// reduction_2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)
        self.sab = SAB()

    def forward(self, x):
        x_d = self.conv1x1_1(x)
        x1 = x_d + self.res(x_d)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x_d + x4

        x5 = self.sab(self.conv1x1_2(x4_1))
        x5_1 = x + x5

        return x5_1
