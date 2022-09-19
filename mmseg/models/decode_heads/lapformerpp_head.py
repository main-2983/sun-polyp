import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer


@HEADS.register_module()
class LAPFormerPPHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super(LAPFormerPPHead, self).__init__(
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
        for i in range(num_inputs - 2):
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
        for idx in range(len(_inputs) - 2, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 2:
                x1 = _inputs[idx + 1]
                x2 = _inputs[idx]
            # if not first 2 then cat from prev outs and _inputs
            else:
                x1 = _out
                x2 = _inputs[idx]
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        out = torch.cat(outs, dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)
        out = resize(
            input=out,
            size=_inputs[0].shape[2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners
        )

        # perform identity mapping
        out = _inputs[0] + out

        out = self.cls_seg(out)

        return out
