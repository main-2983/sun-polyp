import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
from mmseg.models.utils import EfficientSELayer, EfficientLayerAttn, ReversedAttention


@HEADS.register_module()
class MLPeSEHead(BaseDecodeHead):
    def __init__(self,
                 interpolate_mode='bilinear',
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.lateral_attns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.lateral_attns.append(
                EfficientSELayer(
                    channels=self.in_channels[i]
                )
            )
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.layer_attn = EfficientLayerAttn(
            in_channels=self.channels * num_inputs,
            groups=num_inputs
        )

        self.reversed_attn = ReversedAttention()

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
            lateral_attn = self.lateral_attns[idx]
            conv = self.convs[idx]

            feat = lateral_attn(x)
            feat = conv(feat)

            outs.append(
                resize(
                    input=feat,
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = torch.cat(outs, dim=1)
        out = self.layer_attn(out)
        out = self.reversed_attn(out)
        out = self.fusion_conv(out)

        out = self.cls_seg(out)

        return out
