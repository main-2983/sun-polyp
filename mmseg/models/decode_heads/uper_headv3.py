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
from .lib.fem import Uncertain_Boundary

# psp feature instead of last input

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

    

        # self.fuse_feature = BiFPN(self.in_channels, self.channels)
        self.fuse_feature = RCFPN(self.in_channels, self.channels, 5)
        
        
        self.mlp_slow = MLP_OSA(in_channels=self.in_channels, channels=self.channels)

        
        self.reverse_attn = ReverseAttention(
            self.channels,
            1,
            self.conv_cfg, self.norm_cfg, self.act_cfg
        )
        
        self.u_boundary = Uncertain_Boundary(self.channels, self.channels)
        
        self.cls = nn.ModuleList()
        for i in range(4):
            cls_module = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.channels, 1, kernel_size=1))
            self.cls.append(cls_module)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output
    
    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)

        # build top-down path 3, 2, 1
        feats = self.fuse_feature(inputs)
        # # fpn_outs[-2] = fpn_outs[-1]
        # outs = self.mlp_slow(fpn_outs[:-1])
        # out = outs[-1]

        # ## edge attention
        # feats = self.reverse_attn(out, out)
        # feats = self.u_boundary(feats)


        output = self.cls_seg(feats)

        return [output]
