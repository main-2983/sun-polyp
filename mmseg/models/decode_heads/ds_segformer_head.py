import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class SegformerHeadDS(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 num_aux_convs=2,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.aux_cls_seg = nn.ModuleList()
        for i in range(num_inputs):
            aux_convs = []
            for i in range(num_aux_convs):
                aux_convs.append(
                    ConvModule(
                        in_channels=self.channels,
                        out_channels=self.channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
            aux_convs.append(nn.Conv2d(self.channels, 1, 1))
            self.aux_cls_seg.append(nn.Sequential(*aux_convs))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = []
        # lead head prediction
        out = self.fusion_conv(torch.cat(_outs, dim=1))
        out = self.cls_seg(out)
        outs.append(out)
        # aux head prediction
        if self.training:
            for i, _out in enumerate(_outs):
                aux_pred = self.aux_cls_seg[i](_out)
                outs.append(aux_pred)

        return outs


@HEADS.register_module()
class SegformerHeadDS_v2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 num_aux_convs=2,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        aux_convs = []
        for i in range(num_aux_convs):
            chn = self.channels * num_inputs if i == 0 else self.channels
            aux_convs.append(
                ConvModule(
                    in_channels=chn,
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        aux_convs.append(nn.Conv2d(self.channels, 1, 1))
        self.aux_cls_seg = nn.Sequential(*aux_convs)

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = []
        # lead head prediction
        _outs = torch.cat(_outs, dim=1)
        out = self.fusion_conv(_outs)
        out = self.cls_seg(out)
        outs.append(out)
        # aux head prediction
        if self.training:
            aux_pred = self.aux_cls_seg(_outs)
            outs.append(aux_pred)

        return outs