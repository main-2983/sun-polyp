# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize

from .rcfpn import RCFPN
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .lib.attention import *
from .lib.bifpn import BiFPN
from mmseg.models.utils.se_layer import *
from torch import nn
from .lib.mlp_osa import MLP_OSA


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
            self.in_channels[-1],
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.fpn_bottleneck_3 = ConvModule(
            self.channels * 2,
            self.channels, 
            kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


        # self.fuse_feature = BiFPN(self.in_channels, self.channels)
        self.fuse_feature = RCFPN(self.in_channels, self.channels, 4)
        
        
        self.mlp_slow = MLP_OSA(in_channels=self.in_channels, channels=self.channels)


        self.layer_attn = LayerAttention(
            self.channels,
            groups=len(self.in_channels), la_down_rate=8
        )
        
        self.reverse_attn = ReverseAttention(
            self.channels,
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

        inputs = self._transform_inputs(inputs)
        inputs[-1] = self.psp_forward(inputs)
        
        # build top-down path 3, 2, 1
        fpn_outs = self.fuse_feature(inputs)

        for i in fpn_outs:
            print(i.shape)
        out = self.mlp_slow(fpn_outs)


        ## semantic attention 
        fpn_outs_semantic = self.layer_attn(out)
        
        ## edge attention
        fpn_outs_edge = self.reverse_attn(out, out)


        feats = torch.cat([fpn_outs_edge, fpn_outs_semantic], dim=1)

        feats = self.fpn_bottleneck_3(feats)
        
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
