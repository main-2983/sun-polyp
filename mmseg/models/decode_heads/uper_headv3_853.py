# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


from torch import nn



class LayerAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 groups,conv_cfg, norm_cfg, act_cfg, la_down_rate=8):
        super(LayerAttention, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.groups,
                1
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
            kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        # self.ra_conv = ConvModule(
        #     self.in_channels , self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        # )
    def forward(self, input, mul_op):
        
        out = self.fpn_bottleneck(input)
        out = -1*(torch.sigmoid(out)) + 1
        out = out.expand(-1, self.in_channels, -1, -1).mul(mul_op)
        # out = self.ra_conv(out)
        return out


@HEADS.register_module()
class UPerHeadV3(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHeadV3, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                kernel_size=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck_1 = ConvModule(
            self.channels*8,
            self.channels*4,
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        
        self.fpn_bottleneck_2 = ConvModule(
            self.channels*4,
            self.channels * 2, 
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.fpn_bottleneck_3 = ConvModule(
            self.channels * 2,
            self.channels, 
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


        
        
        self.linear_projections = nn.ModuleList()
        for i in range(len(self.in_channels) - 1, 0, -1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * (i+1),
                    out_channels=self.channels* (i+1),
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    conv_cfg=self.conv_cfg
                )
            )

        self.layer_attn = LayerAttention(
            self.channels*4,
            groups=len(self.in_channels), conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, la_down_rate=2
        )
        
        self.reverse_attn = ReverseAttention(
            self.channels*4,
            1,
            self.conv_cfg, self.norm_cfg, self.act_cfg
        )


        

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path 3, 2, 1
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs 0, 1, 2
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
        
        # 3, 2, 1
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        # slow concatenate
        out = torch.empty(
            fpn_outs[0].shape
        )
        for idx in range(len(fpn_outs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx-1]
            # cat first 2 from _inputs
            if idx == len(fpn_outs) - 1:
                x1 = fpn_outs[idx]
                x2 = fpn_outs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = out
                x2 = fpn_outs[idx - 1]
                
            x = torch.cat([x1, x2], dim=1)
            out = linear_prj(x)
            
            
        ## semantic attention 
        fpn_outs_semantic = self.layer_attn(out)
        
        ## edge attention
        fpn_outs_edge = self.reverse_attn(out, out)


        feats = torch.cat([fpn_outs_edge, fpn_outs_semantic], dim=1)
        feats = self.fpn_bottleneck_1(feats)
        feats = self.fpn_bottleneck_2(feats)
        feats = self.fpn_bottleneck_3(feats)
        
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
