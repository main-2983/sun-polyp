import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from mmseg.models.utils.scale import Scale
from .psp_head import PPM
from .da_head import DAM
import torch.nn.functional as F
from lib.conv_layer import Conv, BNPReLU
from lib.axial_atten import AA_kernel
from lib.context_module import CFPModule

#lapformer origin
@HEADS.register_module()
class LAPFormerHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # reduce channel dims and local emphasis
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
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.se_module = SELayer(
            # channels=self.channels * num_inputs
            channels=self.channels * (num_inputs)
        )
        self.fusion_conv = ConvModule(
            # in_channels=self.channels * num_inputs
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = [_inputs[-1]]
        # outs = []
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
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)

        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)
        return out

#lapformer remove concat with high level features 1/32
@HEADS.register_module()
class LAPFormerHead_remove(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # reduce channel dims and local emphasis
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
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.se_module = SELayer(
            # channels=self.channels * num_inputs
            channels=self.channels * (num_inputs - 1)
        )
        self.fusion_conv = ConvModule(
            # in_channels=self.channels * num_inputs
            in_channels=self.channels * (num_inputs - 1),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        # outs = [_inputs[-1]]
        outs = []
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
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)

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
        # forward ppm
        inputs = list(inputs)
        inputs[-1] = self.forward_ppm(inputs)
        inputs[-1] = resize(input=inputs[-1],
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
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


#lapformer remove concat with high level features 1/32 + PPM
@HEADS.register_module()
class LAPFormerHead_remove_PPM(BaseDecodeHead):
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
            channels=self.channels * (num_inputs - 1)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs - 1),
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
        # forward ppm
        inputs = list(inputs)
        inputs[-1] = self.forward_ppm(inputs)
        inputs[-1] = resize(input=inputs[-1],
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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

#remove concat
@HEADS.register_module()
class LAPFormerHead_removeconcat_PPM(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # forward ppm
        inputs = list(inputs)
        inputs[-1] = self.forward_ppm(inputs)
        inputs[-1] = resize(input=inputs[-1],
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = []
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        _inputs.append(inputs[-1])
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out


#Model 1: [1/32, 1/4, 1/8, 1/16] => cat(x[i], x[i - 1])
@HEADS.register_module()
class LAPFormerHead_new(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out

#Model 2: cat(linear(cat(1/16, 1/4)), FSM_out)
@HEADS.register_module()
class LAPFormerHead_new_2(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-2], out], dim=1)
        out = self.cls_seg_2(out)

        return out

#Model 3: cat(FRM_1/4, FSM_out)
@HEADS.register_module()
class LAPFormerHead_new_3(BaseDecodeHead):
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
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
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        FRM_14 = _inputs[1]
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([FRM_14, out], dim=1)
        out = self.cls_seg_2(out)

        return out


#Model 4: addition a cat(1/32, 1/16)
@HEADS.register_module()
class LAPFormerHead_new_4(BaseDecodeHead):
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
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)


        # _inputs: 1/32, 1/4, 1/8, 1/16 => 1/4, 1/8, 1/16, 1/32
        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 2, -1, -1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.insert(0, 
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            
            if idx == 0: #cat with 1/32
                x1 = _out
                x2 = _inputs[-1]
            elif idx == len(_inputs) - 1: # cat first 2 from _inputs
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out















#Model 5: Using Segformer head as neck for Lapformer
@HEADS.register_module()
class LAPFormerHead_new_5(BaseDecodeHead):
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
        

        self.linear_projections_2 = nn.ModuleList()
        for i in range(num_inputs - 1):
            self.linear_projections_2.append(
                ConvModule(
                    in_channels=self.channels * 2,
                    out_channels=self.channels,
                    kernel_size=1,
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

    def forward_ppm(self, inputs):
        # apply PPM on 1/32
        ppm_out = self.ppm(inputs[-1])
        ppm_out.append(inputs[-1])
        ppm_out = torch.cat(ppm_out, dim=1)
        ppm_out = self.bottleneck(ppm_out)
        return ppm_out

    def forward(self, inputs, inputs_2):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            linear_prj_2 = self.linear_projections_2[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]

            _out = torch.add(x1, x2)
            # _out = linear_prj(x)
            _out = torch.cat([_out, inputs_2], dim=1)
            _out = linear_prj_2(_out)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.add(outs[-1], out)
        out = self.cls_seg(out)

        return out






#Model 6: Remove concat 1/16 and 1/8 at segformer_head
@HEADS.register_module()
class LAPFormerHead_new_6(BaseDecodeHead):
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
        

        # self.linear_projections_2 = nn.ModuleList()
        # for i in range(num_inputs - 1):
        #     self.linear_projections_2.append(
        #         ConvModule(
        #             in_channels=self.channels * 2,
        #             out_channels=self.channels,
        #             kernel_size=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))

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

    def forward_ppm(self, inputs):
        # apply PPM on 1/32
        ppm_out = self.ppm(inputs[-1])
        ppm_out.append(inputs[-1])
        ppm_out = torch.cat(ppm_out, dim=1)
        ppm_out = self.bottleneck(ppm_out)
        return ppm_out

    def forward(self, inputs, inputs_2):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            # linear_prj = self.linear_projections[idx - 1]
            linear_prj_2 = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]

            _out = torch.add(x1, x2)
            # _out = linear_prj(x)
            _out = torch.cat([_out, inputs_2], dim=1)
            _out = linear_prj_2(_out)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.add(outs[-1], out)
        out = self.cls_seg(out)

        return out




#Model 7: extend kernel with every scale
@HEADS.register_module()
class LAPFormerHead_new_7(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 pool_scales=(1, 2, 3, 6),
                 ppm_chans=128,
                 dialation_rate=3,
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
                    dilation=dialation_rate,
                    padding=dialation_rate,
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
        # forward ppm
        inputs = list(inputs)
        ppm_output = self.forward_ppm(inputs)
        inputs[-1] = resize(input=ppm_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)

        _inputs = [inputs[-1]]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
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
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out



#Model 8: Remove PPM 
@HEADS.register_module()
class LAPFormerHead_new_7_2(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 pool_scales=(1, 2, 3, 6),
                 ppm_chans=128,
                 dialation_rate=3,
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
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=3,
                    dilation=dialation_rate,
                    padding=dialation_rate,
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
        # forward ppm
        inputs = list(inputs)

        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            
            if idx == 0: #cat with 1/32
                x1 = _out
                x2 = _inputs[-1]
            elif idx == len(_inputs) - 1: # cat first 2 from _inputs
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out



#Model 9
@HEADS.register_module()
class LAPFormerHead_new_9(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 pool_scales=(1, 2, 3, 6),
                 ppm_chans=128,
                 dialation_rate=3,
                 bottleneck_ksize=1,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        print(kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # reduce channel dims and local emphasis
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
        # forward ppm
        inputs = list(inputs)

        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            
            if idx == 0: #cat with 1/32
                x1 = _out
                x2 = _inputs[-1]
            elif idx == len(_inputs) - 1: # cat first 2 from _inputs
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out


#Model 10: using attention modual instead FRM normal
@HEADS.register_module()
class LAPFormerHead_new_10(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        print(kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = nn.ModuleList()
        for i in range(num_inputs):
            self.DAM.append(DAM(
                in_channels_DAM=self.in_channels[i],
                channels_DAM=self.channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ))

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

    # def forward_DAM(self, inputs):
    #     output = self.DAM(inputs)
    #     return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)

        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            DAM = self.DAM[idx]
            _inputs.append(
                resize(
                    input=DAM(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, -1, -1):
            linear_prj = self.linear_projections[idx]
            
            if idx == 0: #cat with 1/32
                x1 = _out
                x2 = _inputs[-1]
            elif idx == len(_inputs) - 1: # cat first 2 from _inputs
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out

#Model 12: concat the same LAPFormerHead_new
@HEADS.register_module()
class LAPFormerHead_new_12(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        print(kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = nn.ModuleList()
        for i in range(num_inputs):
            self.DAM.append(DAM(
                in_channels_DAM=self.in_channels[i],
                channels_DAM=self.channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ))

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

    # def forward_DAM(self, inputs):
    #     output = self.DAM(inputs)
    #     return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)

        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            DAM = self.DAM[idx]
            _inputs.append(
                resize(
                    input=DAM(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]

            if idx == 1:
                x1 = _out
                x2 = _inputs[-1]
            # cat first 2 from _inputs
            elif idx == len(_inputs) - 1:
                x1 = _inputs[idx - 1]
                x2 = _inputs[idx - 2]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 2]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out

#Model 13: using attention modual instead last FRM
@HEADS.register_module()
class LAPFormerHead_new_13(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        print(kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = DAM(
            in_channels_DAM=self.in_channels[-1],
            channels_DAM=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
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
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    # def forward_DAM(self, inputs):
    #     output = self.DAM(inputs)
    #     return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)

        DAM_output = self.DAM(inputs[-1])
        DAM_output = resize(input=DAM_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)


        _inputs = [DAM_output]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            # print("len _inputs: ", len(_inputs), "idx: ", idx)
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out


#Model 16: Add scale của mai vào 3 FRM.
@HEADS.register_module()
class LAPFormerHead_new_16(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 scale_pos='residual',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        assert scale_pos in ['residual', 'layer']
        self.scale_pos = scale_pos
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = DAM(
            in_channels_DAM=self.in_channels[-1],
            channels_DAM=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
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
        self.pff_scales = nn.ModuleList()

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
            self.pff_scales.append(
                Scale(channels=self.channels,
                      init_val=1.0 if scale_pos == 'residual' else 1e-5)
            )

        self.se_module = SELayer(
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    # def forward_DAM(self, inputs):
    #     output = self.DAM(inputs)
    #     return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)

        DAM_output = self.DAM(inputs[-1])
        DAM_output = resize(input=DAM_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)


        _inputs = [DAM_output]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            pff_scale = self.pff_scales[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            
            if self.scale_pos == 'residual':
                x = torch.cat([pff_scale(x1), pff_scale(x2)], dim = 1)
            else:
                x = torch.cat([pff_scale(x1), pff_scale(x2)], dim = 1)
            # x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out


#lapformer origin
@HEADS.register_module()
class LAPFormerHead_new_15(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 scale_pos='residual',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.scale_pos = scale_pos
        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = DAM(
            in_channels_DAM=self.in_channels[-1],
            channels_DAM=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
            )

        # reduce channel dims and local emphasis
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

        for i in range(num_inputs - 1):
            self.linear_projections.append(
                ConvModule(
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
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
            # channels=self.channels * num_inputs
            channels=self.channels * (num_inputs)
        )
        self.fusion_conv = ConvModule(
            # in_channels=self.channels * num_inputs
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)

        _inputs = []

        DAM_output = self.DAM(inputs[-1])
        DAM_output = resize(input=DAM_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)


        _inputs = [DAM_output]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        
        _inputs.insert(-1, _inputs.pop(0))
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = [_inputs[-1]]
        # outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            pff_scale = self.pff_scales[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            if self.ops == 'cat':
                if self.scale_pos == 'residual':
                    x = torch.cat([pff_scale(x1), pff_scale(x2)], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)

        # perform identity mapping
        out = outs[-1] + out

        out = self.cls_seg(out)
        return out



@HEADS.register_module()
class LAPFormerHead_new_17(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 ops='cat',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        # reduce channel dims and local emphasis
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
                    in_channels=self.channels * 2 if self.ops == 'cat' else self.channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        self.se_module = SELayer(
            # channels=self.channels * num_inputs
            channels=self.channels * (num_inputs)
        )
        self.fusion_conv = ConvModule(
            # in_channels=self.channels * num_inputs
            in_channels=self.channels * (num_inputs),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        # ColonFormer Head 2
        self.CFP_1 = CFPModule(128, d = 8)
        self.CFP_2 = CFPModule(320, d = 8)
        self.CFP_3 = CFPModule(512, d = 8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(128,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(320,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(512,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = AA_kernel(128,128)
        self.aa_kernel_2 = AA_kernel(320,320)
        self.aa_kernel_3 = AA_kernel(512,512)


    def forward_lapformer(self, inputs):
        inputs = self._transform_inputs(inputs)
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        _inputs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))
        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = [_inputs[-1]]
        # outs = []
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
            if self.ops == 'cat':
                x = torch.cat([x1, x2], dim=1)
            else:
                x = x1 + x2
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)

        # perform identity mapping
        out = outs[-1] + out

        out_lapformer = self.cls_seg(out)
        return out_lapformer

    def forward(self, inputs):
        out_lapformer = self.forward_lapformer(inputs)
        lateral_map_1 = F.interpolate(out_lapformer, scale_factor=4, mode='bilinear')
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(out_lapformer, scale_factor=0.125, mode='bilinear')
        cfp_out_1 = self.CFP_3(inputs[3])
        # cfp_out_1 += x4
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3 += cfp_out_1
        aa_atten_3_o = decoder_2_ra.expand(-1, 512, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) 
        ra_3 = self.ra3_conv2(ra_3) 
        ra_3 = self.ra3_conv3(ra_3) 
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3,scale_factor=32,mode='bilinear')
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_2(inputs[2])
        # cfp_out_2 += x3
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2 += cfp_out_2
        aa_atten_2_o = decoder_3_ra.expand(-1, 320, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) 
        ra_2 = self.ra2_conv2(ra_2) 
        ra_2 = self.ra2_conv3(ra_2) 
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2,scale_factor=16,mode='bilinear')        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(inputs[1])
        # cfp_out_3 += x2
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1 += cfp_out_3
        aa_atten_1_o = decoder_4_ra.expand(-1, 128, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) 
        ra_1 = self.ra1_conv2(ra_1) 
        ra_1 = self.ra1_conv3(ra_1) 
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1,scale_factor=8,mode='bilinear') 
        
        return [lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1]



@HEADS.register_module()
class LAPFormerHead_new_18(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 scale_pos='residual',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        assert scale_pos in ['residual', 'layer']
        self.scale_pos = scale_pos
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.DAM = DAM(
            in_channels_DAM=self.in_channels[-1],
            channels_DAM=self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
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
        self.pff_scales = nn.ModuleList()

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
            self.pff_scales.append(
                Scale(channels=self.channels,
                      init_val=1.0 if scale_pos == 'residual' else 1e-5)
            )

        self.se_module = SELayer(
            channels=self.channels
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    # def forward_DAM(self, inputs):
    #     output = self.DAM(inputs)
    #     return output

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        # forward ppm
        inputs = list(inputs)

        DAM_output = self.DAM(inputs[-1])
        DAM_output = resize(input=DAM_output,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)


        _inputs = [DAM_output]
        for idx in range(len(inputs) - 1):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        # slow concatenate
        _out = torch.empty(
            _inputs[0].shape
        )
        outs = []
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            pff_scale = self.pff_scales[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx]
                x2 = _inputs[idx - 1]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx - 1]
            
            if self.scale_pos == 'residual':
                x = torch.cat([pff_scale(x1), pff_scale(x2)], dim = 1)
            else:
                x = torch.cat([pff_scale(x1), pff_scale(x2)], dim = 1)
            # x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # out = torch.cat(outs, dim=1)
        out = self.se_module(outs[-1])
        out = self.fusion_conv(outs[-1])
        # perform identity mapping
        out = torch.cat([outs[-1], out], dim=1)
        out = self.cls_seg_2(out)

        return out










#remove concat
# @HEADS.register_module()
# class LAPFormerHead_removeconcat_PPM_2(BaseDecodeHead):
#     def __init__(self,
#                  interpolate_mode='bilinear',
#                  pool_scales=(1, 2, 3, 6),
#                  ppm_chans=128,
#                  bottleneck_ksize=1,
#                  **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)

#         assert num_inputs == len(self.in_index)

#         # ppm module and bottleneck
#         self.ppm = PPM(
#             in_channels=self.in_channels[-1],
#             pool_scales=pool_scales,
#             channels=ppm_chans,
#             conv_cfg=None,
#             norm_cfg=None,
#             act_cfg=self.act_cfg,
#             align_corners=self.align_corners
#         )
#         self.bottleneck = ConvModule(
#             in_channels=self.in_channels[-1] + len(pool_scales) * ppm_chans,
#             out_channels=self.channels,
#             kernel_size=bottleneck_ksize,
#             padding=bottleneck_ksize//2,
#             stride=1,
#             norm_cfg=self.norm_cfg
#         )

#         # reduce channel dims and local emphasis
#         self.convs = nn.ModuleList()
#         for i in range(num_inputs - 1):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=3,
#                     padding=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))

#         # feature fusion between adjacent levels
#         self.linear_projections = nn.ModuleList()
#         for i in range(num_inputs - 1):
#             self.linear_projections.append(
#                 ConvModule(
#                     in_channels=self.channels * 2,
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg
#                 )
#             )

#         self.se_module = SELayer(
#             channels=self.channels
#         )
#         self.fusion_conv = ConvModule(
#             in_channels=self.channels,
#             out_channels=self.channels,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)

#     def forward_ppm(self, inputs):
#         # apply PPM on 1/32
#         ppm_out = self.ppm(inputs[-1])
#         ppm_out.append(inputs[-1])
#         ppm_out = torch.cat(ppm_out, dim=1)
#         ppm_out = self.bottleneck(ppm_out)
#         return ppm_out

#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         # forward ppm
#         inputs = list(inputs)
#         inputs[-1] = self.forward_ppm(inputs)
#         inputs[-1] = resize(input=inputs[-1],
#                             size=inputs[0].shape[2:],
#                             mode=self.interpolate_mode,
#                             align_corners=self.align_corners)

#         _inputs = []
#         for idx in range(len(inputs) - 1):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             _inputs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))
#         _inputs.append(inputs[-1])
#         # slow concatenate
#         _out = torch.empty(
#             _inputs[0].shape
#         )
#         outs = []
#         for idx in range(len(_inputs) - 1, 0, -1):
#             linear_prj = self.linear_projections[idx - 1]
#             # cat first 2 from _inputs
#             if idx == len(_inputs) - 1:
#                 x1 = _inputs[idx]
#                 x2 = _inputs[idx - 1]
#             # if not first 2 then cat from prev outs and _inputs
#             else:
#                 x1 = _out
#                 x2 = _inputs[idx - 1]
#             x = torch.cat([x1, x2], dim=1)
#             _out = linear_prj(x)
#             outs.append(_out)

#         # out = torch.cat(outs, dim=1) # cat linear projection
#         # out = self.se_module(outs[-1])
#         # out = self.fusion_conv(outs[-1])
#         # perform identity mapping
#         # out = torch.cat([outs[-1], out], dim=1) # skip connection
#         out = self.cls_seg(outs[-1])
#         return out


# @HEADS.register_module()
# class LAPFormerHead_concat_last_first_PPM(BaseDecodeHead):
#     def __init__(self,
#                  interpolate_mode='bilinear',
#                  pool_scales=(1, 2, 3, 6),
#                  ppm_chans=128,
#                  bottleneck_ksize=1,
#                  **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)

#         self.interpolate_mode = interpolate_mode
#         num_inputs = len(self.in_channels)

#         assert num_inputs == len(self.in_index)
        


#         # ppm module and bottleneck
#         self.ppm = PPM(
#             in_channels=self.in_channels[-1],
#             pool_scales=pool_scales,
#             channels=ppm_chans,
#             conv_cfg=None,
#             norm_cfg=None,
#             act_cfg=self.act_cfg,
#             align_corners=self.align_corners
#         )
#         self.bottleneck = ConvModule(
#             in_channels=self.in_channels[-1] + len(pool_scales) * ppm_chans,
#             out_channels=self.channels,
#             kernel_size=bottleneck_ksize,
#             padding=bottleneck_ksize//2,
#             stride=1,
#             norm_cfg=self.norm_cfg
#         )

#         # reduce channel dims and local emphasis
#         self.convs = nn.ModuleList()
#         for i in range(num_inputs - 1):
#             self.convs.append(
#                 ConvModule(
#                     in_channels=self.in_channels[i],
#                     out_channels=self.channels,
#                     kernel_size=3,
#                     padding=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg))

#         # feature fusion between adjacent levels
#         self.linear_projections = nn.ModuleList()
#         for i in range(num_inputs - 1):
#             self.linear_projections.append(
#                 ConvModule(
#                     in_channels=self.channels * 2,
#                     out_channels=self.channels,
#                     kernel_size=1,
#                     stride=1,
#                     norm_cfg=self.norm_cfg,
#                     act_cfg=self.act_cfg
#                 )
#             )

#         self.se_module = SELayer(
#             channels=self.channels * 2
#         )
#         self.fusion_conv = ConvModule(
#             in_channels=self.channels * 2,
#             out_channels=self.channels,
#             kernel_size=1,
#             norm_cfg=self.norm_cfg)

#     def forward_ppm(self, inputs):
#         # apply PPM on 1/32
#         ppm_out = self.ppm(inputs[-1])
#         ppm_out.append(inputs[-1])
#         ppm_out = torch.cat(ppm_out, dim=1)
#         ppm_out = self.bottleneck(ppm_out)
#         return ppm_out

#     def forward(self, inputs):
#         # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
#         inputs = self._transform_inputs(inputs)
#         # forward ppm
#         inputs = list(inputs)
#         inputs[-1] = self.forward_ppm(inputs)
#         inputs[-1] = resize(input=inputs[-1],
#                             size=inputs[0].shape[2:],
#                             mode=self.interpolate_mode,
#                             align_corners=self.align_corners)

#         _inputs = []
#         for idx in range(len(inputs) - 1):
#             x = inputs[idx]
#             conv = self.convs[idx]
#             _inputs.append(
#                 resize(
#                     input=conv(x),
#                     size=inputs[0].shape[2:],
#                     mode=self.interpolate_mode,
#                     align_corners=self.align_corners))
#         _inputs.append(inputs[-1])
#         # slow concatenate
#         _out = torch.empty(
#             _inputs[0].shape
#         )
#         outs = [_inputs[-1]]
#         for idx in range(len(_inputs) - 1, 0, -1):
#             linear_prj = self.linear_projections[idx - 1]
#             # cat first 2 from _inputs
#             if idx == len(_inputs) - 1:
#                 x1 = _inputs[idx]
#                 x2 = _inputs[idx - 1]
#             # if not first 2 then cat from prev outs and _inputs
#             else:
#                 x1 = _out
#                 x2 = _inputs[idx - 1]
#             x = torch.cat([x1, x2], dim=1)
#             _out = linear_prj(x)

#             if idx == 1:
#               outs.append(_out)
#         out = torch.cat(outs, dim=1)
#         out = self.se_module(out)
#         out = self.fusion_conv(out)
#         # perform identity mapping
#         out = torch.cat([outs[-1], out], dim=1)
#         out = self.cls_seg_2(out)

#         return out