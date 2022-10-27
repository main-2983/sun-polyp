import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils.receptive_field_block import RFB
from mmseg.models.utils.partial_decoder import PartialDecoder
from mmseg.models.utils.reversed_attention import ReversedAttention

@HEADS.register_module()
class SPraNetHead(BaseDecodeHead):
    def __init__(self,
                 rfb_channels=32,
                 revattn_channels=[256, 64, 64],
                 revattn_kernel=[5, 3, 3],
                 revattn_convs=[3, 2, 2],
                 **kwargs):
        super(SPraNetHead, self).__init__(input_transform='multiple_select',
                                         **kwargs)

        num_inputs = len(self.in_channels)
        self.interpolate_mode = 'bilinear'
        assert num_inputs == len(self.in_index) ==\
               len(revattn_convs) == len(revattn_channels) == len(revattn_kernel)

        self.rfbs = nn.ModuleList()
        for i in range(num_inputs):
            self.rfbs.append(
                RFB(self.in_channels[i],
                    out_channels=rfb_channels)
            )
        self.partial_decoder = PartialDecoder(rfb_channels)
        self.global_pred = nn.Conv2d(num_inputs * rfb_channels, 1, 1)
        self.reversed_attns = nn.ModuleList()
        for i in range(num_inputs):
            self.reversed_attns.append(
                ReversedAttention(in_channels=self.in_channels[::-1][i],
                                  channels=revattn_channels[i],
                                  kernel_size=revattn_kernel[i],
                                  num_convs=revattn_convs[i])
            )
        self.aux_pred = nn.ModuleList()
        for i in range(num_inputs):
            self.aux_pred.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=revattn_channels[i],
                        out_channels=revattn_channels[i],
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True)
                    ),
                    ConvModule(
                        in_channels=revattn_channels[i],
                        out_channels=1,
                        kernel_size=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        act_cfg=None
                    )
                )
            )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        rfb_outs = []
        for i in range(len(self.rfbs)):
            rfb_outs.append(
                self.rfbs[i](inputs[i])
            )
        rfb_outs = rfb_outs[::-1]
        partial_decoder_out = self.partial_decoder(*rfb_outs)
        global_map = self.global_pred(partial_decoder_out) # (size/8)

        outs = [global_map]
        for i in range(len(self.reversed_attns)):
            y = resize(input=outs[-1],
                       size=inputs[len(inputs) - (i+1)].shape[2:],
                       mode=self.interpolate_mode)
            x = inputs[len(inputs) - (i+1)]
            _x, y = self.reversed_attns[i](x, y)
            _x = self.aux_pred[i](_x)
            out = _x + y
            outs.append(out) # (/32 -> /16 -> /8)
        # reverse order to match with SunSegmentor output requirements
        outs = outs[::-1]

        return outs