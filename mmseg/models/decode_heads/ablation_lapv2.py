import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from .lap_headv2 import LAPHead_v2_34


# ExFuse, remove cat all, cat -> add and scale, add SE
# Add DWConv to first stage
@HEADS.register_module()
class ALAPv2_34(LAPHead_v2_34):
    def __init__(self,
                 dw_kernel_size=3,
                 **kwargs):
        super().__init__(**kwargs)

        self.convs[3] = ConvModule(
                        in_channels=self.in_channels[3],
                        out_channels=self.channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
        self.convs[0] = DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[0],
                        out_channels=self.channels,
                        kernel_size=dw_kernel_size,
                        padding=dw_kernel_size//2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            inputs[idx] = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners)

        outs = []
        for idx in range(len(inputs) -1, -1, -1):
            linear_prj = self.linear_prj[idx]
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            else:
                x1 = out
                x2 = inputs[idx - 1]
            x = x1 + x2
            out = linear_prj(x)
            outs.append(out)

        out = outs[-1]
        out = self.se_module(out)
        out = self.cls_seg(out)

        return out