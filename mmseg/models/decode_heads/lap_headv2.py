import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from mmseg.models.utils.pyramid_pooling_attention import PyramidPoolingAttention, PyramidPoolingAttentionv2
from mmseg.models.utils.strip_attention import *
from mmseg.models.utils.triplet_attention import *
from mmseg.models.utils.cbam_bam import BAMSpatial
from mmseg.models.utils.scale import Scale
from .psp_head import PPM


# Remove 1/4 in concatenation, add in later
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
            channels=self.channels * (num_inputs - 1)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs - 1),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = [] # 1/4, 1/8, 1/8, 1/8
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            feat = conv(x)
            if idx > 0:
                feat = resize(
                    input=feat,
                    size=inputs[1].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            _inputs.append(feat)

        # progressive feature fusion
        _out = torch.empty(
            _inputs[1].shape
        )
        outs = [_inputs[-1]]
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            if idx == 1:
                x1 = resize(
                    input=x1,
                    size=x2.shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs[:-1], dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        out = resize(
            input=out,
            size=_inputs[0].shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )

        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out


# Add PPM module to last scale
@HEADS.register_module()
class LAPHead_v2_2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 pool_scales=(1, 2, 3, 6),
                 ppm_chans=128,
                 bottleneck_ksize=1,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # ppm module and bottleneck
        self.ppm = PPM(
            in_channels=self.in_channels[-1],
            pool_scales=pool_scales,
            channels=ppm_chans,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners
        )
        self.bottleneck = ConvModule(
            in_channels=self.in_channels[-1] + len(pool_scales) * ppm_chans,
            out_channels=self.channels,
            kernel_size=bottleneck_ksize,
            padding=bottleneck_ksize//2,
            stride=1,
            norm_cfg=self.norm_cfg
        )

        # reduce channel dims and local emphasis
        self.convs = nn.ModuleList()
        for i in range(num_inputs - 1):
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

    def forward_ppm(self, inputs):
        # apply PPM on 1/32
        ppm_out = self.ppm(inputs[-1])
        ppm_out.append(inputs[-1])
        ppm_out = torch.cat(ppm_out, dim=1)
        ppm_out = self.bottleneck(ppm_out)
        return ppm_out

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            if idx != (len(inputs) - 1):
                conv = self.convs[idx]
                inputs[idx] = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            else:
                inputs[idx] = resize(
                    input=self.forward_ppm(inputs),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
        # slow concatenate
        outs = [inputs[-1]]
        for idx in range(len(inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(inputs) - 1:
                x1 = inputs[idx]
                x2 = inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
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


# New concatenation, move concatenation one level down
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


# Replace 1/32 FRM with PPA
@HEADS.register_module()
class LAPHead_v2_5(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ppa_act=None,
                 ppa_scales=(1, 2, 3, 6),
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # Add PPA
        self.ppa = nn.Sequential(
            PyramidPoolingAttention(
                channels=self.in_channels[-1],
                act_cfg=ppa_act,
                pool_scales=ppa_scales
            ),
            ConvModule(
                in_channels=self.in_channels[-1],
                out_channels=self.channels,
                kernel_size=1,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg
            )
        )

        self.convs = nn.ModuleList()
        for i in range(num_inputs - 1):
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

    def forward_ppa(self, inputs):
        # Forward 1/32 to PPA
        x = inputs[-1]
        out = self.ppa(x)
        return out

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            if idx != (len(inputs) - 1):
                conv = self.convs[idx]
                inputs[idx] = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            else:
                inputs[idx] = resize(
                    input=self.forward_ppa(inputs),
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


# Replace 1/32 FRM with PPM
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

        # Add PPM
        self.ppm = PPM(
            in_channels=self.in_channels[-1],
            pool_scales=(1, 2, 3, 6),
            channels=128,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            in_channels=self.in_channels[-1] + len([1, 2, 3, 6]) * 128,
            out_channels=self.channels,
            kernel_size=1,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)

        self.convs = nn.ModuleList()
        for i in range(num_inputs - 1):
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

    def forward_ppm(self, inputs):
        # Forward 1/32 to PPA
        x = inputs[-1]
        outs = [x]
        outs.extend(self.ppm(x))
        out = torch.cat(outs, dim=1)
        out = self.bottleneck(out)
        return out

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            if idx != (len(inputs) - 1):
                conv = self.convs[idx]
                inputs[idx] = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            else:
                inputs[idx] = resize(
                    input=self.forward_ppm(inputs),
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


# New concatenation, move concatenation one level down
# Add PPA
@HEADS.register_module()
class LAPHead_v2_7(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ppa_act=None,
                 ppa_scales=(1, 2, 3, 6),
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            **kwargs
        )
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # Add PPA
        self.ppa = nn.Sequential(
            PyramidPoolingAttention(
                channels=self.in_channels[-1],
                act_cfg=ppa_act,
                pool_scales=ppa_scales
            ),
            ConvModule(
                in_channels=self.in_channels[-1],
                out_channels=self.channels,
                kernel_size=1,
                act_cfg=self.act_cfg,
                norm_cfg=self.norm_cfg
            )
        )

        self.convs = nn.ModuleList()
        for i in range(num_inputs - 1):
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

    def forward_ppa(self, inputs):
        # Forward 1/32 to PPA
        x = inputs[-1]
        out = self.ppa(x)
        return out

    def forward(self, inputs):
        # inputs: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        for idx in range(len(inputs)):
            x = inputs[idx]
            if idx != (len(inputs) - 1):
                conv = self.convs[idx]
                inputs[idx] = resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            else:
                inputs[idx] = resize(
                    input=self.forward_ppa(inputs),
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


# v2_4,
# Replace FRM with MSStripConvAttn
@HEADS.register_module()
class LAPHead_v2_8(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 strip_kernels=(3, 5, 7),
                 strip_act=None,
                 strip_norm=None,
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
                MSStripConvAttn(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_sizes=strip_kernels,
                    act_cfg=strip_act,
                    norm_cfg=strip_norm))

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
        assert scale_pos in ['residual', 'prev']
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
                Scale(channels=self.channels)
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
