import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer
from .psp_head import PPM
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
class LAPFormerHead_PPM_RemConcat_new(BaseDecodeHead):
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
class LAPFormerHead_PPM_RemConcat_new_2(BaseDecodeHead):
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
class LAPFormerHead_PPM_RemConcat_new_3(BaseDecodeHead):
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
class LAPFormerHead_PPM_RemConcat_new_4(BaseDecodeHead):
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
class LAPFormerHead_PPM_RemConcat_new_5(BaseDecodeHead):
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