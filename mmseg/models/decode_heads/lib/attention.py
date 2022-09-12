# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv
import math
from mmcv.cnn import ConvModule
from mmseg.models.utils import *
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        projected_query = self.query_conv(x).reshape(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).reshape(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).reshape(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
    




class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels//2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels//2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels//4)
        self.linear_2 = nn.Linear(self.in_channels//4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        
        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats


class LayerAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 groups, la_down_rate=8):
        super(LayerAttention, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                kernel_size=3, padding=1
            ),
            nn.Sigmoid()
        )
        
        # self.la_conv = ConvModule(self.in_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def forward(self, x):
        b, c, h, w = x.shape

        avg_feat = nn.AdaptiveAvgPool2d(1)(x)           # average pooling like every fucking attention do
        weight = self.layer_attention(avg_feat)         # make weight of shape (b, groups, 1, 1)

        x = x.view(b, self.groups, c // self.groups, h, w)
        weight = weight.view(b, self.groups, 1, 1, 1)
        _x = x.clone()
        for group in range(self.groups):
            _x[:, group] = x[:, group] * weight[:, group]

        _x = _x.view(b, c, h, w)
        # _x = self.la_conv(_x)

        return _x


        
class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.fpn_bottleneck = ConvModule(
            self.in_channels, self.out_channels,
            kernel_size=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.ra_conv = ConvModule(
            self.in_channels , self.in_channels, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
    def forward(self, input, mul_op):
        
        out = self.fpn_bottleneck(input)
        out = -1*(torch.sigmoid(out)) + 1
        out = out.expand(-1, self.in_channels, -1, -1).mul(mul_op)
        out = self.ra_conv(out)
        return out
    


class EfficientSELayer(nn.Module):
    def __init__(self,
                 channels,
                 conv_cfg=None):
        super(EfficientSELayer, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.attn_weight = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                act_cfg=None
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.attn_weight(out)

        return x * out