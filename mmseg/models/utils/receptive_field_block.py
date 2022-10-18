import torch
import torch.nn as nn
from mmcv.cnn import ConvModule


class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.branch0 = ConvModule(in_channels, out_channels, 1,
                                  act_cfg=None, norm_cfg=norm_cfg)
        self.branch1 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 3),
                       padding=(0, 1), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (3, 1),
                       padding=(1, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=3,
                       dilation=3, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 5),
                       padding=(0, 2), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (5, 1),
                       padding=(2, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=5,
                       dilation=5, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.branch3 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 7),
                       padding=(0, 3), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (7, 1),
                       padding=(3, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=7,
                       dilation=7, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.fusion_conv = ConvModule(
            out_channels * 4, out_channels, 3, padding=1, act_cfg=None)
        self.residual_conv = ConvModule(
            in_channels, out_channels, 1, act_cfg=None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.fusion_conv(torch.cat([x0, x1, x2, x3], dim=1))
        x = self.relu(x_cat + self.residual_conv(x))
        return x