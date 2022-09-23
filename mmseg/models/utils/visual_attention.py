import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


""" A set of Convolutional Attention from VAN's authors """
class LargeKernelAttn(nn.Module):
    def __init__(self,
                 channels):
        super(LargeKernelAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        return x * weight


class MultiScaleConvAttn(nn.Module):
    def __init__(self,
                 channels):
        super(MultiScaleConvAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.scale_7 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=channels
            )
        )
        self.scale_11 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                groups=channels
            )
        )
        self.scale_21 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                groups=channels
            ),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                groups=channels
            )
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        base_weight = self.dwconv(x)
        weight1 = self.scale_7(base_weight)
        weight2 = self.scale_11(base_weight)
        weight3 = self.scale_21(base_weight)
        weight = base_weight + weight1 + weight2 + weight3
        weight = self.pwconv(weight)

        return x * weight


class MultiScaleLocalAttn(nn.Module):
    def __init__(self,
                 channels):
        super(MultiScaleLocalAttn, self).__init__()
        self.dwconv_3 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels
        )
        self.dwconv_5 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        weight1 = self.dwconv_3(x)
        weight2 = self.dwconv_5(x)
        weight = weight1 + weight2
        weight = self.pwconv(weight)

        return x * weight
