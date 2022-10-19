from turtle import forward
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from .conv import Conv


class FeatureEnhanceModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        
        self.conv1 = Conv(in_channels, in_channels, kernel_size=3, dilation=1, groups=in_channels, relu=True)
        self.conv2 = Conv(in_channels, in_channels, kernel_size=3, dilation=3, groups=in_channels, relu=True)
        self.conv3 = Conv(in_channels, in_channels, kernel_size=3, dilation=5, groups=in_channels, relu=True)
        
        self.fusion = Conv(in_channels * 3, out_channels, kernel_size=1, relu=True)
        
    def forward(self, inp):
        inp1 = self.conv1(inp)
        inp2 = self.conv2(inp)
        inp3 = self.conv3(inp)
        
        output = self.fusion(torch.cat(inp1, inp2, inp3), dim=1)
        
        return output
        