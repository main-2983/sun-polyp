import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import ReversedAttention


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReceptiveFieldBlock, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.branch0 = ConvModule(in_channels, out_channels, 1,
                                  act_cfg=None, norm_cfg=norm_cfg)
        self.branch1 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 3),
                       padding=(0, 1), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (3, 1),
                       padding=(1, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=3,
                       dilation=3, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 5),
                       padding=(0, 2), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (5, 1),
                       padding=(2, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=5,
                       dilation=5, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.branch3 = nn.Sequential(
            ConvModule(in_channels, out_channels, 1,
                       act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (1, 7),
                       padding=(0, 3), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, (7, 1),
                       padding=(3, 0), act_cfg=None, norm_cfg=norm_cfg),
            ConvModule(out_channels, out_channels, 3, padding=7,
                       dilation=7, act_cfg=None, norm_cfg=norm_cfg)
        )
        self.fusion_conv = ConvModule(
            out_channels * 4, out_channels, 3, padding=1, act_cfg=None)
        self.residual_conv = ConvModule(
            in_channels, out_channels, 1, act_cfg=None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.fusion_conv(torch.cat([x0, x1, x2, x3], dim=1))
        x = self.relu(x_cat + self.residual_conv(x))
        return x


class PartialDecoder(nn.Module):
    """ This implementation is a little different from original PraNet
    in PraNet, the last Conv layer is actually a layer to generate prediction,
    we remove it in this implementation and move the prediction layer to decode head
    """
    def __init__(self, channels):
        super(PartialDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample4 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample5 = ConvModule(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            act_cfg=None)

        self.conv_cat2 = ConvModule(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_cat3 = ConvModule(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            act_cfg=None)

        self.conv4 = ConvModule(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            act_cfg=None)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_cat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_cat3(x3_2)

        x = self.conv4(x3_2)

        return x


class AuxHead(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_convs=0,
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
        if num_convs == 0:
            self.aux_convs = nn.Identity()
            self.aux_cls_seg = ConvModule(in_channels, 1, 1,
                                          act_cfg=None,
                                          norm_cfg=norm_cfg)
        else:
            self.aux_convs = nn.Sequential(*aux_convs)
            self.aux_cls_seg = ConvModule(channels, 1, 1,
                                          act_cfg=None,
                                          norm_cfg=norm_cfg)

    def forward(self, x):
        aux_out = self.aux_convs(x)
        aux_out = self.aux_cls_seg(aux_out)

        return aux_out


@HEADS.register_module()
class PraNetHead(BaseDecodeHead):
    def __init__(self,
                 rfb_channels=32,
                 revattn_channels=[256, 64, 64],
                 revattn_kernel=[5, 3, 3],
                 revattn_convs=[3, 2, 2],
                 num_aux_convs=0,
                 aux_channels=None,
                 **kwargs):
        super(PraNetHead, self).__init__(input_transform='multiple_select',
                                         **kwargs)

        num_inputs = len(self.in_channels)
        self.interpolate_mode = 'bilinear'
        assert num_inputs == len(self.in_index) ==\
               len(revattn_convs) == len(revattn_channels) == len(revattn_kernel)

        self.rfbs = nn.ModuleList()
        for i in range(num_inputs):
            self.rfbs.append(
                ReceptiveFieldBlock(self.in_channels[i],
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
                AuxHead(
                    in_channels=revattn_channels[i],
                    channels=aux_channels,
                    num_convs=num_aux_convs
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
