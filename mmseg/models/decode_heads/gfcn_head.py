import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class GFCNHead(BaseDecodeHead):
    def __init__(self,
                 input_transform='multiple_select',
                 interpolate_mode='bilinear',
                 num_convs=2,
                 **kwargs):
        super(GFCNHead, self).__init__(input_transform=input_transform, **kwargs)
        self.interpolate_mode = interpolate_mode
        self.num_inputs = len(self.in_channels)
        self.convs = nn.ModuleList()
        for i in range(self.num_inputs):
            inter_conv = []
            inter_conv.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))
            for i in range(num_convs - 1):
                inter_conv.append(
                    ConvModule(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        conv_cfg=self.conv_cfg))
            self.convs.append(nn.Sequential(*inter_conv))
        self.fusion_conv = ConvModule(
            in_channels=self.num_inputs * self.channels,
            out_channels=self.channels,
            kernel_size=1
        )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for i in range(len(inputs)):
            conv = self.convs[i]
            out = conv(inputs[i])
            outs.append(
                resize(input=out,
                       size=inputs[0].shape[2:],
                       mode=self.interpolate_mode,
                       align_corners=self.align_corners))
        outs = torch.cat(outs, dim=1)
        outs = self.fusion_conv(outs)
        out = self.cls_seg(outs)

        return out
