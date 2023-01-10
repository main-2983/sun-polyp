import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

__all__ = [
    'TripletAttention'
]

class ZPool(nn.Module):
    def forward(self, x):
        _max = torch.max(x, dim=1, keepdim=True)[0]
        _mean = torch.mean(x, dim=1, keepdim=True)
        return torch.cat([_max, _mean], dim=1)


class AttentionModule(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 act_cfg=dict(
                     type='Sigmoid'
                 )):
        super(AttentionModule, self).__init__()
        self.compress = ZPool()
        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            act_cfg=act_cfg
        )

    def forward(self, x):
        x_compress = self.compress(x)
        weight = self.conv(x_compress)
        return x * weight


class TripletAttention(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 act_cfg=dict(
                     type='Sigmoid'
                 )):
        super(TripletAttention, self).__init__()
        self.cw = AttentionModule(kernel_size, act_cfg)
        self.hc = AttentionModule(kernel_size, act_cfg)
        self.hw = AttentionModule(kernel_size, act_cfg)

    def forward(self, x):
        # x: B, C, H, W
        x_cw = x.permute(0, 2, 1, 3).contiguous() # B, H, C, W
        x_hc = x.permute(0, 3, 2, 1).contiguous() # B, W, H, C
        x_hw = x

        x_out1 = self.cw(x_cw)
        x_out1 = x_out1.permute(0, 2, 1, 3) # B, C, H, W

        x_out2 = self.hc(x_hc)
        x_out2 = x_out2.permute(0, 3, 2, 1) # B, C, H, W

        x_out3 = self.hw(x_hw)

        x_out = 1/3 * (x_out1 + x_out2 + x_out3)

        return x_out


class TripletWeight(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 act_cfg=dict(
                     type='Sigmoid'
                 )):
        super(TripletWeight, self).__init__()
        self.compress = ZPool()
        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            act_cfg=act_cfg
        )

    def forward(self, x):
        x_compress = self.compress(x)
        weight = self.conv(x_compress)
        return weight


class TripletAttention_v2(nn.Module):
    def __init__(self,
                 kernel_size=7,
                 attn_cfg=dict(
                     type='ReLU'
                 ),
                 act_cfg=None):
        super(TripletAttention_v2, self).__init__()
        self.cw = TripletWeight(kernel_size, attn_cfg)
        self.hc = TripletWeight(kernel_size, attn_cfg)
        self.hw = TripletWeight(kernel_size, attn_cfg)
        self.bottleneck = ConvModule(
            in_channels=1,
            out_channels=1
        )

    def forward(self, x):
        # x: B, C, H, W
        x_cw = x.permute(0, 2, 1, 3).contiguous()  # B, H, C, W
        x_hc = x.permute(0, 3, 2, 1).contiguous()  # B, W, H, C
        x_hw = x

        x_out1 = self.cw(x_cw)
        x_out1 = x_out1.permute(0, 2, 1, 3)  # B, C, H, W

        x_out2 = self.hc(x_hc)
        x_out2 = x_out2.permute(0, 3, 2, 1)  # B, C, H, W

        x_out3 = self.hw(x_hw)

        x_out = 1 / 3 * (x_out1 + x_out2 + x_out3)

        return x_out
