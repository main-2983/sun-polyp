import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .segformer_head import SegformerHead


class AuxHead(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=2,
                 kernel_size=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(AuxHead, self).__init__()
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=kernel_size//2,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg
                )
            )
        self.convs = nn.Sequential(*convs)
        self.aux_seg = nn.Conv2d(in_channels=channels,
                                 out_channels=num_classes,
                                 kernel_size=1)

    def forward(self, x):
        out = self.convs(x)
        out = self.aux_seg(out)
        return out


@HEADS.register_module()
class CascadeSegformerHead(SegformerHead):
    def __init__(self,
                 num_aux_convs=2,
                 aux_kernel=3,
                 **kwargs):
        super(CascadeSegformerHead, self).__init__(**kwargs)
        self.fusion_conv = None # Remove fusion_conv
        # Construct Aux Heads
        self.aux_heads = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.aux_heads.append(
                AuxHead(in_channels=self.channels,
                        channels=self.channels,
                        num_classes=self.num_classes,
                        num_convs=num_aux_convs,
                        kernel_size=aux_kernel,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs) # 1/4, 1/8, 1/16, 1/32
        _outs = [] # 1/32, 1/16, 1/8, 1/4
        for idx in range(len(inputs) - 1, -1, -1):
            x = inputs[idx]
            conv = self.convs[idx]
            out = conv(x)
            if idx < len(inputs) - 1:
                out = out + resize(_outs[-1],
                                   scale_factor=2,
                                   mode=self.interpolate_mode,
                                   align_corners=self.align_corners)
            _outs.append(out)

        # Forward Aux Heads
        outs = []
        if self.training:
            for i in range(len(self.aux_heads)):
                featmaps = _outs[i]
                out = self.aux_heads[i](featmaps)
                outs.append(out)
        # Forward Main Head
        out = self.cls_seg(_outs[-1])
        outs.append(out)
        # Format output
        outs = outs[::-1]

        return outs
