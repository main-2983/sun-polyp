import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from mmseg.models.utils.pyramid_pooling_attention import *
from mmseg.models.utils.strip_attention import *
from mmseg.models.utils.strip_conv import *
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
        return outs[-2]
        # perform identity mapping
        out = outs[-2] + out

        out = self.cls_seg(out)

        return out


# Add one more level of concatenation
@HEADS.register_module()
class LAPHead_v2_2(BaseDecodeHead):
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


# ExFuse (3 scale)
# Change skip connection from (1/8 + 1/4) -> (1/32 + 1/4)
@HEADS.register_module()
class LAPHead_v2_3(BaseDecodeHead):
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
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (4 scale)
# Change skip connection from (1/8 + 1/4) -> (1/32 + 1/4)
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
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (3 scale), (1/32 + /1/4) in skip
# Drop concat all
@HEADS.register_module()
class LAPHead_v2_5(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # rearrange inputs to 1/32, 1/4, 1/8, 1/16
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

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (4 scale), (1/32 + /1/4) in skip
# Drop concat all
@HEADS.register_module()
class LAPHead_v2_6(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (4 scale), (1/32 + /1/4) in skip, drop concat all
# change cat -> add in PFF
@HEADS.register_module()
class LAPHead_v2_7(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (3 scale), (1/32 + /1/4) in skip, drop concat all
# change cat -> add in PFF
@HEADS.register_module()
class LAPHead_v2_8(BaseDecodeHead):
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
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.se_module = SELayer(
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # rearrange inputs to 1/32, 1/4, 1/8, 1/16
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
            x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# ExFuse (3 scale), drop concat all
# drop skip
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # rearrange inputs to 1/32, 1/4, 1/8, 1/16
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

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)

        out = self.cls_seg(out)

        return out


# ExFuse (3 scale), (1/32 + /1/4) in skip, drop concat all
# Attempt to use both skip: use with Scale init to 0
@HEADS.register_module()
class LAPHead_v2_10(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 scale_val=1.0,
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.scales = nn.ModuleList([
            Scale(self.channels, scale_val),
            Scale(self.channels, scale_val)
        ])

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
        # rearrange inputs to 1/32, 1/4, 1/8, 1/16
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

        out = outs[-1]
        out = self.se_module(out)
        out = self.fusion_conv(out)
        # perform identity mapping
        out = self.scales[0](outs[-1]) + self.scales[1](outs[-2]) + out

        out = self.cls_seg(out)

        return out


#################################################################
@HEADS.register_module()
class LAPHead_v2_20(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
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
                    kernel_size=1 if i < num_inputs - 1 else 3,
                    stride=1,
                    padding=0 if i < num_inputs - 1 else 1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out


# SlowCat
@HEADS.register_module()
class LAPHead_v2_21(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
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
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.linear_prj = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_prj.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

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

        outs = [inputs[-1]]
        for idx in range(len(inputs) -1, 0, -1):
            linear_prj = self.linear_prj[idx - 1]
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            else:
                x1 = out
                x2 = inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            out = linear_prj(x)
            outs.append(out)

        outs = torch.cat(outs, dim=1)
        out = self.fusion_conv(outs)

        out = self.cls_seg(out)

        return out


# Add PPSA to 1/32
@HEADS.register_module()
class LAPHead_v2_22(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ppsa_pools=(1, 2, 3, 6),
                 ppsa_act=dict(
                     type='Softmax',
                     dim=-1
                 ),
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            if i != num_inputs - 1:
                self.convs.append(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            else:
                self.convs.append(
                    PyramidPoolingSelfAttention(
                        in_channels=self.in_channels[i],
                        out_channels=self.channels,
                        pool_scales=ppsa_pools,
                        act_cfg=ppsa_act
                    )
                )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out


# Add PPSA to 1/32
@HEADS.register_module()
class LAPHead_v2_23(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ppsa_pools=(1, 2, 3, 6),
                 ppsa_act=dict(
                     type='Softmax',
                     dim=-1
                 ),
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            if i != num_inputs - 1:
                self.convs.append(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            else:
                self.convs.append(
                    PyramidPoolingSelfAttentionv2(
                        in_channels=self.in_channels[i],
                        out_channels=self.channels,
                        down_rate=4,
                        pool_scales=ppsa_pools,
                        act=ppsa_act
                    )
                )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out


# ExFuse
@HEADS.register_module()
class LAPHead_v2_24(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
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
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.linear_prj = nn.ModuleList()
        for i in range(num_inputs):
            self.linear_prj.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

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
            x = torch.cat([x1, x2], dim=1)
            out = linear_prj(x)
            outs.append(out)

        outs = torch.cat(outs, dim=1)
        out = self.fusion_conv(outs)

        out = self.cls_seg(out)

        return out
