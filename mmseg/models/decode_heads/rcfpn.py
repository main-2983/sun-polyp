import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmcv.cnn import ConvModule, xavier_init, constant_init


class FusionNode(nn.Module):

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 with_out_conv=True,
                 out_conv_cfg=None,
                 out_norm_cfg=None,
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
            self.spatial_weight = nn.Conv2d(
                in_channels * 2, 1, kernel_size=3, padding=1, bias=True)
            self.temp = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=True)
            for m in self.spatial_weight.modules():
                if isinstance(m, nn.Conv2d):
                    constant_init(m, 0)

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

        if op_num > 2:
            self.pre_fusion = ConvModule(
                out_channels,
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
        weight1 = self.gap(x1)
        weight2 = self.gap(x2)
        if self.upsample_attn:
            upsample_weight = (
                self.temp * channel**(-0.5) *
                self.spatial_weight(torch.cat((x1, x2), dim=1)))
            upsample_weight = F.softmax(
                upsample_weight.reshape(batch, 1, -1), dim=-1).reshape(
                    batch, 1, height, width) * height * width
            x2 = upsample_weight * x2
        weight = torch.cat((weight1, weight2), dim=1)
        weight = self.weight[0](weight)
        weight = torch.sigmoid(weight)
        result = weight * x1 + (1 - weight) * x2
        if self.op_num == 3:
            x3 = x[2]
            x1 = self.pre_fusion(result)
            # x1 = result
            
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


class RCFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 norm_cfg=None,
                 out_conv_cfg=None):
        super(RCFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)  # num of input feature levels
        self.num_outs = num_outs  # num of output feature levels
        self.norm_cfg = norm_cfg

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        # add lateral connections
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1, padding=0,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.lateral_convs.append(l_conv)




        self.RevFP = nn.ModuleDict()
        self.RevFP['p7'] = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=2,
            upsample_attn=False)

        self.RevFP['p6'] = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=3)

        self.RevFP['p5'] = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=3)

        self.RevFP['p4'] = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=3)

        self.RevFP['p3'] = FusionNode(
            in_channels=out_channels,
            out_channels=out_channels,
            out_conv_cfg=out_conv_cfg,
            out_norm_cfg=norm_cfg,
            op_num=2)
        
            
    def init_weights(self):
        """Initialize the weights of module."""
        for m in self.lateral_convs.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        # build P3-P5
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        c3, c4, c5, c6 = feats
        c7 = inputs[-1]        
        # fixed order 
        p3 = self.RevFP['p3']([c3, c4], out_size=c3.shape[-2:])
        p4 = self.RevFP['p4']([c4, c5, p3], out_size=c4.shape[-2:])
        p5 = self.RevFP['p5']([c5, c6, p4], out_size=c5.shape[-2:])
        p6 = self.RevFP['p6']([c6, c7, p5], out_size=c6.shape[-2:])
        p7 = self.RevFP['p7']([c7, p6], out_size=c7.shape[-2:])

        return [p3, p4, p5, p6, p7]