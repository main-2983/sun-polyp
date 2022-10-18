import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class GFCNHeadDS(BaseDecodeHead):
    def __init__(self,
                 input_transform='multiple_select',
                 interpolate_mode='bilinear',
                 num_convs=2,
                 **kwargs):
        super(GFCNHeadDS, self).__init__(input_transform=input_transform, **kwargs)
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
        self.conv_seg_auxs = nn.ModuleList()
        for i in range(self.num_inputs):
            self.conv_seg_auxs.append(
                nn.Conv2d(self.channels, self.num_classes, 1)
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        _outs = []
        for i in range(len(inputs)):
            conv = self.convs[i]
            out = conv(inputs[i])
            _outs.append(
                resize(input=out,
                       size=inputs[0].shape[2:],
                       mode=self.interpolate_mode,
                       align_corners=self.align_corners))
        outs = []
        cat_outs = torch.cat(_outs, dim=1)
        cat_outs = self.fusion_conv(cat_outs)
        out = self.cls_seg(cat_outs)
        outs.append(out)
        # aux head forward
        if self.training:
            for i in range(len(_outs)):
                aux_out = self.conv_seg_auxs[i](_outs[i])
                outs.append(aux_out)
        return outs


@HEADS.register_module()
class GFCNHeadDS_v2(BaseDecodeHead):
    def __init__(self,
                 input_transform='multiple_select',
                 interpolate_mode='bilinear',
                 num_convs=2,
                 **kwargs):
        super(GFCNHeadDS_v2, self).__init__(input_transform=input_transform, **kwargs)
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
        self.conv_seg_auxs = nn.ModuleList()
        for i in range(self.num_inputs):
            self.conv_seg_auxs.append(
                nn.Conv2d(self.channels, self.num_classes, 1)
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        _outs = []
        noresize_outs = []
        for i in range(len(inputs)):
            conv = self.convs[i]
            out = conv(inputs[i])
            noresize_outs.append(out)
            _outs.append(
                resize(input=out,
                       size=inputs[0].shape[2:],
                       mode=self.interpolate_mode,
                       align_corners=self.align_corners))
        outs = []
        cat_outs = torch.cat(_outs, dim=1)
        cat_outs = self.fusion_conv(cat_outs)
        out = self.cls_seg(cat_outs)
        outs.append(out)
        # aux head forward
        if self.training:
            for i in range(len(noresize_outs)):
                aux_out = self.conv_seg_auxs[i](noresize_outs[i])
                outs.append(aux_out)
        return outs