import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from mmseg.models.utils.scale import Scale


class AuxHead(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_convs=2,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(AuxHead, self).__init__()
        aux_convs = []
        for i in range(num_convs):
            chn = in_channels if i == 0 else channels
            aux_convs.append(
                ConvModule(
                    in_channels=chn,
                    out_channels=channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
        self.aux_convs = nn.Sequential(*aux_convs)
        self.aux_cls_seg = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        aux_out = self.aux_convs(x)
        aux_out = self.aux_cls_seg(aux_out)

        return aux_out



@HEADS.register_module()
class LAPHead_v2_11_DS(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # feature fusion between adjacent levels
        self.linear_projections = nn.ModuleList()
        self.pff_scales = nn.ModuleList()
        for i in range(num_inputs):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.pff_scales.append(
                nn.ModuleList([
                    Scale(channels=self.channels,
                          init_val=1.0),
                    Scale(channels=self.channels,
                          init_val=1e-2)
                ])
            )

        self.aux_head = AuxHead(
            in_channels=self.channels * num_inputs,
            channels=self.channels
        )

        self.se_module = SELayer(
            channels=self.channels * num_inputs
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            inputs[idx] = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

        # outs: 1/32 + 1/16, 1/16 + 1/8, 1/8 + 1/4, 1/4 + 1/32
        outs = []
        for idx in range(len(inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            # cat first 2 from inputs
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            # if not first 2 then cat from prev outs and inputs
            else:
                x1 = _out
                x2 = inputs[idx - 1]
            # Layer Scale + Res Scale
            x = self.pff_scales[idx][0](x1) + self.pff_scales[idx][1](x2)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + out

        all_outs = []
        # lead head pred
        out = self.cls_seg(out)
        all_outs.append(out)
        # aux head pred
        if self.training:
            aux_out = self.aux_head(torch.cat(outs, dim=1))
            all_outs.append(aux_out)

        return all_outs


@HEADS.register_module()
class LAPHead_v2_11_DS_2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # feature fusion between adjacent levels
        self.linear_projections = nn.ModuleList()
        self.pff_scales = nn.ModuleList()
        for i in range(num_inputs):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.pff_scales.append(
                nn.ModuleList([
                    Scale(channels=self.channels,
                          init_val=1.0),
                    Scale(channels=self.channels,
                          init_val=1e-2)
                ])
            )

        self.se_module = SELayer(
            channels=self.channels * num_inputs
        )

        self.aux_head = AuxHead(
            in_channels=self.channels * num_inputs,
            channels=self.channels
        )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            inputs[idx] = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

        # outs: 1/32 + 1/16, 1/16 + 1/8, 1/8 + 1/4, 1/4 + 1/32
        outs = []
        for idx in range(len(inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            # cat first 2 from inputs
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            # if not first 2 then cat from prev outs and inputs
            else:
                x1 = _out
                x2 = inputs[idx - 1]
            # Layer Scale + Res Scale
            x = self.pff_scales[idx][0](x1) + self.pff_scales[idx][1](x2)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        # keep this to use in aux head
        _out = self.se_module(out)
        out = self.fusion_conv(_out)
        # perform identity mapping
        out = outs[-2] + out

        all_outs = []
        # lead head pred
        out = self.cls_seg(out)
        all_outs.append(out)
        # aux head pred
        if self.training:
            aux_out = self.aux_head(_out)
            all_outs.append(aux_out)

        return all_outs


@HEADS.register_module()
class LAPHead_v2_11_DS_3(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # feature fusion between adjacent levels
        self.linear_projections = nn.ModuleList()
        self.pff_scales = nn.ModuleList()
        for i in range(num_inputs):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            self.pff_scales.append(
                nn.ModuleList([
                    Scale(channels=self.channels,
                          init_val=1.0),
                    Scale(channels=self.channels,
                          init_val=1e-2)
                ])
            )

        self.se_module = SELayer(
            channels=self.channels * num_inputs
        )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.aux_head = AuxHead(
            in_channels=self.channels,
            channels=self.channels
        )

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            inputs[idx] = resize(
                input=conv(x),
                size=inputs[0].shape[2:],
                mode=self.interpolate_mode,
                align_corners=self.align_corners
            )

        # outs: 1/32 + 1/16, 1/16 + 1/8, 1/8 + 1/4, 1/4 + 1/32
        outs = []
        for idx in range(len(inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            # cat first 2 from inputs
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            # if not first 2 then cat from prev outs and inputs
            else:
                x1 = _out
                x2 = inputs[idx - 1]
            # Layer Scale + Res Scale
            x = self.pff_scales[idx][0](x1) + self.pff_scales[idx][1](x2)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        # keep this to use in aux head
        out = self.se_module(out)
        _out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + _out

        all_outs = []
        # lead head pred
        out = self.cls_seg(out)
        all_outs.append(out)
        # aux head pred
        if self.training:
            aux_out = self.aux_head(_out)
            all_outs.append(aux_out)

        return all_outs
