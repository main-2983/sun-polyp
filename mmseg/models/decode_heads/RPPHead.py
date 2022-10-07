import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
import torch
from .decode_head import BaseDecodeHead
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, constant_init
from .lib.axial_attention import AA_kernel
torch.autograd.set_detect_anomaly(True)



class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output
class ScaleBranch(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super(ScaleBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatialPooling = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.scalePooling = nn.AdaptiveAvgPool2d(1)
        self.channel_agg = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.trans = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        x = self.spatialPooling(x.permute(0, 2, 1, 3, 4)).reshape(
            x.size(0), x.size(2), 4, 1)
        batch, channel, height, width = x.size()
        channel_context = self.channel_agg(x)
        channel_context = channel_context.view(batch, 1, height * width)
        channel_context = F.softmax(channel_context, dim=-1)
        channel_context = channel_context * height * width
        channel_context = channel_context.view(batch, 1, height, width)
        context = self.scalePooling(x * channel_context)
        context = self.trans(context).unsqueeze(0)
        return context


class SpatialBranch(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super(SpatialBranch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scalePooling = nn.AvgPool3d((4, 1, 1))
        self.spatialPooling = nn.AdaptiveAvgPool2d(1)
        self.channel_agg = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.trans = nn.Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        x = self.scalePooling(x.permute(0, 2, 1, 3, 4)).squeeze(2)
        batch, channel, height, width = x.size()
        channel_context = self.channel_agg(x)
        channel_context = channel_context.view(batch, 1, height * width)
        channel_context = F.softmax(channel_context, dim=-1)
        channel_context = channel_context * height * width
        channel_context = channel_context.view(batch, 1, height, width)
        context = self.spatialPooling(x * channel_context)
        context = self.trans(context).unsqueeze(0)
        return context
class FusionNode(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_conv_cfg=None,
                 out_norm_cfg=dict(type='BN', requires_grad=True),
                 out_conv_order=('act', 'conv', 'norm'),
                 upsample_mode='bilinear',
                 op_num=2,
                 upsample_attn=True):
        super(FusionNode, self).__init__()
        assert op_num == 2 or op_num == 3
        self.with_out_conv = with_out_conv
        self.upsample_mode = upsample_mode
        self.op_num = op_num
        self.upsample_attn = upsample_attn
        act_cfg = None
        self.act_cfg = act_cfg

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True)
        constant_init(self.weight, 0)

        self.fusion = nn.ModuleList()
        for i in range(op_num-1):
            self.fusion.append(ConvModule(in_channels*2, in_channels, kernel_size=1, norm_cfg=out_norm_cfg, order=out_conv_order))
            for m in self.fusion[-1].modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')

        if self.upsample_attn:
            self.spatial_weight = nn.Conv2d(
                in_channels * 2, 1, kernel_size=3, padding=1, bias=True)
            self.temp = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=True)
            for m in self.spatial_weight.modules():
                if isinstance(m, nn.Conv2d):
                    constant_init(m, 0)

        # if self.with_out_conv:
        #     self.post_fusion = ConvModule(
        #         out_channels,
        #         out_channels,
        #         kernel_size=3,
        #         padding=1,
        #         conv_cfg=out_conv_cfg,
        #         norm_cfg=out_norm_cfg,
        #         order=('act', 'conv', 'norm'))
        # if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
        #     for m in self.post_fusion.modules():
        #         if isinstance(m, nn.Conv2d):
        #             xavier_init(m, distribution='uniform')

        if op_num > 2:
            self.post_fusion = ConvModule(
                out_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=out_conv_cfg,
                norm_cfg=out_norm_cfg,
                order=('act', 'conv', 'norm'))
            if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
                for m in self.post_fusion.modules():
                    if isinstance(m, nn.Conv2d):
                        xavier_init(m, distribution='uniform')

    def dynamicFusion(self, x):
        x1, x2 = x[0], x[1]
        batch, channel, height, width = x1.size()

        if self.upsample_attn:
            upsample_weight = (
                self.temp * channel**(-0.5) *
                self.spatial_weight(torch.cat((x1, x2), dim=1)))
            upsample_weight = F.softmax(
                upsample_weight.reshape(batch, 1, -1), dim=-1).reshape(
                    batch, 1, height, width) * height * width
            x2 = upsample_weight * x2
        result = self.fusion[0](torch.cat([x1,x2], dim=1))
        if self.op_num == 3:
            x3 = x[2]
            f_1 = result
            f_2 = self.fusion[1](torch.cat([x2, x3], dim=1))

            weight1 = self.gap(f_1)
            weight3 = self.gap(f_2)
            weight = torch.cat((weight1, weight3), dim=1)
            weight = self.weight(weight)
            weight = torch.sigmoid(weight)
            result = weight * f_1 + (1 - weight) * f_2
            if self.with_out_conv:
                result = self.post_fusion(result)
        return result

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)
        else:
            _, _, h, w = x.size()
            x = F.max_pool2d(
                F.pad(x, [0, w % 2, 0, h % 2], 'replicate'), (2, 2))
            return x

    def forward(self, x, out_size=None):
        inputs = []
        for feat in x:
            inputs.append(self._resize(feat, out_size))
        return self.dynamicFusion(inputs)

    
@HEADS.register_module()
class RPPHead(BaseDecodeHead):

    def __init__(self,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_conv_cfg=None, **kwargs):
        super(RPPHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.num_ins = len(self.in_channels)  # num of input feature levels
        self.norm_cfg = norm_cfg

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Sequential(
                ConvModule(
                self.in_channels[i],
                self.channels,
                kernel_size=3, padding=1,
                norm_cfg=norm_cfg),
            )
            self.lateral_convs.append(l_conv)

        self.RevFP = nn.ModuleDict()

        self.spatialContext = SpatialBranch(self.channels,
                                            self.channels)
        self.scaleContext = ScaleBranch(self.channels, self.channels)
        

        self.RevFP['p5'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=2, upsample_attn=False)

        self.RevFP['p4'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=2, upsample_attn=False)

        self.RevFP['p3'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=2, upsample_attn=False)
        
        self.bn_relu = BNPReLU(self.channels * 4)
        
        self.fusion_conv = ConvModule(self.channels * 4, self.channels,
                kernel_size=1, padding=0, norm_cfg=norm_cfg)

        self.aa_module = AA_kernel(self.channels, self.channels)

    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] < size:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            _, _, h, w = x.size()
            x = F.max_pool2d(
                F.pad(x, [0, w % 2, 0, h % 2], 'replicate'), (2, 2))
            return x


    def forward(self, inputs):
        """Forward function."""
        # build P3-P5
        inputs = self._transform_inputs(inputs)
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        c3, c4, c5, c6 = feats
                
        c4 = self._resize(c4, size=c3.shape[-2:])
        c5 = self._resize(c5, size=c3.shape[-2:])
        c6 = self._resize(c6, size=c3.shape[-2:])
        # fixed order 
        output = torch.cat([c3.unsqueeze(1), c4.unsqueeze(1), c5.unsqueeze(1), c6.unsqueeze(1)], dim=1)  # B*4*C*H*W
        
        output = output.permute(1, 0, 2, 3, 4) + self.spatialContext(output) + self.scaleContext(output)    
        
        list_p = [c3, c4, c5, c6]
        for i in range(4):
            list_p[i] = list_p[i] + output[i]
        c3, c4, c5, c6 = list_p

        aa_atten = self.aa_module(p3)
        
        p5 = self.RevFP['p3']([c5, c6], out_size=c3.shape[-2:])
        p4 = self.RevFP['p4']([c4, p5], out_size=c3.shape[-2:])
        p3 = self.RevFP['p5']([c3, p4], out_size=c3.shape[-2:])


        
        z3 = p3 + aa_atten
        z4 = c4 + p3
        z5 = c5 + p4
        z6 = c6 + p5
        
        output = self.bn_relu(torch.cat([z3, z4, z5, z6], dim=1))
        output = self.fusion_conv(output)
        
        output = self.cls_seg(output)
        return [output]
