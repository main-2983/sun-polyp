import torch
import torch.nn as nn

from .se_layer import SELayer

class ReversedAttention(nn.Module):
    def __init__(self):
        super(ReversedAttention, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        out_feat = -1 * torch.sigmoid(max_feat) + 1
        out_feat = out_feat.expand([-1, c, -1, -1])

        feat = out_feat * x

        return feat


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
