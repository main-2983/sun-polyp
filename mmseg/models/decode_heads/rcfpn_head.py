import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
import torch
from .decode_head import BaseDecodeHead
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, constant_init
from .lib.attention import PSABlock
torch.autograd.set_detect_anomaly(True)



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.ReLU(nIn)

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

class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)
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

        self.weight = nn.ModuleList()
        self.gap = nn.AdaptiveAvgPool2d(1)
        for i in range(op_num - 1):
            self.weight.append(
                nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True))
            constant_init(self.weight[-1], 0)

        if self.upsample_attn:
            self.attn_gate = Attention_block(out_channels, out_channels, out_channels)

        if self.with_out_conv:
            self.post_fusion = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=out_conv_cfg,
                norm_cfg=out_norm_cfg,
                order=('act', 'conv', 'norm'))
        if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
            for m in self.post_fusion.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')

        self.pre_fusion = ConvModule(
            out_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=out_conv_cfg,
            norm_cfg=out_norm_cfg,
            order=('act', 'conv', 'norm'))
        if out_conv_cfg is None or out_conv_cfg['type'] == 'Conv2d':
            for m in self.pre_fusion.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')

    def dynamicFusion(self, x):
        x1, x2 = x[0], x[1]
        batch, channel, height, width = x1.size()

        if self.upsample_attn:
            x2 = self.attn_gate(x1,x2)
        result = self.pre_fusion(torch.cat([x1, x2], dim=1))
        
        if self.op_num == 3:
            x3 = x[2]
            x1 = result
            weight1 = self.gap(x1)
            weight3 = self.gap(x3)
            weight = torch.cat((weight1, weight3), dim=1)
            weight = self.weight[1](weight)
            weight = torch.sigmoid(weight)
            result = weight * x1 + (1 - weight) * x3
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
class RPFNHead(BaseDecodeHead):

    def __init__(self,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_conv_cfg=None, **kwargs):
        super(RPFNHead, self).__init__(input_transform='multiple_select', **kwargs)
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
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg,
                order=('act', 'conv', 'norm'))
            self.lateral_convs.append(l_conv)

        self.RevFP = nn.ModuleDict()
        self.RevFP['p6'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            op_num=2, upsample_attn=True)

        self.RevFP['p5'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            op_num=2, upsample_attn=True)

        self.RevFP['p4'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            op_num=2, upsample_attn=True)

        self.RevFP['p3'] = FusionNode(
            in_channels=self.channels,
            out_channels=self.channels,
            op_num=2, upsample_attn=True)
        
        self.psa1 = PSABlock(self.channels, self.channels)
        self.psa2 = PSABlock(self.channels, self.channels)
        self.psa3 = PSABlock(self.channels, self.channels)
        self.psa4 = PSABlock(self.channels, self.channels)
        
        self.fusion_conv = ConvModule(self.channels * 4, self.channels,
                kernel_size=1, padding=0, norm_cfg=norm_cfg)

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
        
        c3 = self.psa1(c3)
        c4 = self.psa2(c4)
        c5 = self.psa3(c5)
        c6 = self.psa4(c6)
        

        # fixed order 
        p3 = self.RevFP['p3']([c3, c4], out_size=c3.shape[-2:])
        p4 = self.RevFP['p4']([c4, c5, p3], out_size=c4.shape[-2:])
        p5 = self.RevFP['p5']([c5, c6, p4], out_size=c5.shape[-2:])
        p6 = self.RevFP['p6']([c6, p5], out_size=c6.shape[-2:])

        p4 = self._resize(p4, size=c3.shape[-2:])
        p5 = self._resize(p5, size=c3.shape[-2:])
        p6 = self._resize(p6, size=c3.shape[-2:])
        
        # p6 = self.psa1(p6, p6)
        # p5 = self.psa2(p5, p6)
        # p4 = self.psa3(p4, p5)
        # p3 = self.psa4(p3, p4)
        

        
        
        output = self.fusion_conv(torch.cat([p3,p4,p5,p6], dim=1))

        

        output = self.cls_seg(output)
        return [output]