import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from mmseg.models.utils.pyramid_pooling_attention import PyramidPoolingAttention
from mmseg.models.utils.strip_attention import *
from mmseg.models.utils.strip_conv import *
from mmseg.models.utils.triplet_attention import *
from mmseg.models.utils.large_kernel import *
from mmseg.models.utils.cbam_bam import BAMSpatial
from mmseg.models.utils.scale import Scale
from mmseg.models.utils.osa_module import *
from mmseg.models.utils.spatial_attn import *
from .psp_head import PPM


# New concatenation, move concatenation one level down
@HEADS.register_module()
class LAPHead_v2_1(BaseDecodeHead):
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
        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
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

        # new concatenation order:
        # 1/16 + 1/8 -> 1/8 + 1/4 -> 1/4 + 1/32
        # rearange inputs to 1/32, 1/4, 1/8, 1/16
        inputs = inputs[-1:] + inputs[0:-1]
        # 1/16, 1/16 + 1/8, 1/8 + 1/4, 1/4 + 1/32
        outs = [inputs[-1]]
        # idx: 3, 2, 1
        for idx in range(len(inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from inputs
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            # if not first 2 then cat from prev outs and inputs
            else:
                x1 = _out
                x2 = inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + out

        out = self.cls_seg(out)

        return out


# Add one more level of concatenation
@HEADS.register_module()
class LAPHead_v2_4(BaseDecodeHead):
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
        for i in range(num_inputs):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + out

        out = self.cls_seg(out)

        return out


# Change cat -> add in PFF
@HEADS.register_module()
class LAPHead_v2_9(BaseDecodeHead):
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
            x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + out

        out = self.cls_seg(out)

        return out


# Change cat -> add in PFF but with Scale
# Scale can be add in 'residual' or output from prev PFF
@HEADS.register_module()
class LAPHead_v2_20(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 scale_pos='residual',
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        assert scale_pos in ['residual', 'layer']
        self.scale_pos = scale_pos
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
                Scale(channels=self.channels,
                      init_val=1.0 if scale_pos == 'residual' else 1e-5)
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
            if self.scale_pos == 'residual':
                x = self.pff_scales[idx](x1) + x2
            else:
                x = x1 + self.pff_scales[idx](x2)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-2] + out

        out = self.cls_seg(out)

        return out


# Change cat -> add in PFF but with Scale
# Scale is added in both 'residual' and output from prev PFF (ResScale + LayerScale)
@HEADS.register_module()
class LAPHead_v2_11(BaseDecodeHead):
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

        out = self.cls_seg(out)

        return out

