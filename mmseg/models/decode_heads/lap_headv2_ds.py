import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.ops import resize
from .lap_headv2 import LAPHead_v2_34


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
class LAPHead_v2_34_DS(LAPHead_v2_34):
    def __init__(self,
                 aux_channels=256,
                 num_aux_conv=1,
                 **kwargs):
        super(LAPHead_v2_34_DS, self).__init__(**kwargs)
        self.aux_head = AuxHead(in_channels=self.channels,
                                channels=aux_channels,
                                num_convs=num_aux_conv)

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
            x = x1 + x2
            out = linear_prj(x)
            outs.append(out)

        out = outs[-1]
        out = self.se_module(out)
        out = self.cls_seg(out)

        if self.training:
            aux_out = self.aux_head(outs[-2])

        return [out, aux_out] if self.training else out
