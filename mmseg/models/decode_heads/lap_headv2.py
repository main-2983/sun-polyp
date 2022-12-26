import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import SELayer


# Drop 1/4 in concatenation process, use it with add later
@HEADS.register_module()
class LAPHead_v2_1(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

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
            channels=self.channels * (num_inputs-1)
        )
        self.fusion_conv = ConvModule(
            in_channels=self.channels * (num_inputs-1),
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        _inputs = [] # 1/4, 1/8, 1/16, 1/32
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            _inputs.append(conv(x))

        # slow concatenate
        outs = [resize(_inputs[-1],
                       size=inputs[1].shape[2:],
                       mode=self.interpolate_mode,
                       align_corners=self.align_corners)] # 1/8, 1/8, 1/8, 1/4
        # reversed order
        for idx in range(len(_inputs) - 1, 0, -1):
            linear_prj = self.linear_projections[idx - 1]
            # cat first 2 from _inputs
            if idx == len(_inputs) - 1:
                x1 = _inputs[idx] # scale_i
                x2 = _inputs[idx - 1] # scale_i * 2
            # if not first 2 then cat from prev out (_out) and _inputs
            else:
                x1 = _out # scale_i
                x2 = _inputs[idx - 1] # scale_i * 2
            if idx == 1:
                # resize to 1/4
                x1 = resize(x1,
                            size=inputs[0].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)
            else:
                # resize to 1/8
                x1 = resize(x1,
                            size=inputs[1].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)
                x2 = resize(x2,
                            size=inputs[1].shape[2:],
                            mode=self.interpolate_mode,
                            align_corners=self.align_corners)
            x = torch.cat([x1, x2], dim=1)
            _out = linear_prj(x)
            outs.append(_out)

        # cat all 1/8
        out = torch.cat(outs[:-1], dim=1)
        out = self.se_module(out)
        out = self.fusion_conv(out)

        # perform identity mapping
        out = resize(input=out,
                     size=inputs[0].shape[2:],
                     mode=self.interpolate_mode,
                     align_corners=self.align_corners)
        out = outs[-1] + out

        out = self.cls_seg(out)

        return out