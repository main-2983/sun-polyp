import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .se_layer import SELayer

class ReversedAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_stacked=3):
        super(ReversedAttention, self).__init__()
        self.bottleneck = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=None
        )
        convs = []
        for i in range(num_stacked):
            convs.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        convs.append(
            ConvModule(
                in_channels=out_channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
                act_cfg=None
            )
        )
        self.convs = nn.Sequential(*convs)

    def forward(self, x, y):
        b, c, h, w = x.shape

        bottleneck = self.bottleneck(x)
        rev_weight = -1 * torch.sigmoid(y) + 1
        rev_weight = rev_weight.expand([-1, c, -1, -1])
        out_feat = bottleneck * rev_weight
        out_feat = self.convs(out_feat)

        return y + out_feat

class ReversedAttention_v2(nn.Module):
    def __init__(self, channels):
        super(ReversedAttention_v2, self).__init__()
        self.channel_attn = SELayer(channels=channels)

    def forward(self, x):
        b, c, h, w = x.shape

        feat = self.channel_attn(x)
        max_feat, _ = torch.max(feat, dim=1, keepdim=True)
        out_feat = -1 * torch.sigmoid(max_feat) + 1
        out_feat = out_feat.expand([-1, c, -1, -1])

        out = out_feat * x

        return out
