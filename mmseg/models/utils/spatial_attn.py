import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.cnn.utils import normal_init, constant_init


__all__ = [
    'AvgSpatialAttn', 'DisentangledSpatialSA',
    'AttentionGate'
]


class AvgSpatial(nn.Module):
    def __init__(self, dim=1):
        super(AvgSpatial, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=True)


class AvgSpatialAttn(nn.Module):
    """
    This module use spatial mean to squeeze the feature maps into 2D plane of shape (B, 1, H, W)
    Then use a large kernel conv to generate attention weight
    The attention weight is then later expanded to perform mul with feature maps
    """
    def __init__(self,
                 channels,
                 kernel_size=7,
                 norm_cfg=None,
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(AvgSpatialAttn, self).__init__()
        self.avg = AvgSpatial()
        self.conv = ConvModule(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            padding=kernel_size//2
        )
        self.channels = channels

    def forward(self, x):
        weight = self.avg(x)
        weight = self.conv(weight)
        weight = weight.expand(-1, self.channels, -1, -1)
        return weight * x


class DisentangledSpatialSA(nn.Module):
    """
    This is a re-implementation of pair-wise term in Disentangled Non-Local Network
    References: https://arxiv.org/pdf/2006.06668.pdf
    """
    def __init__(self,
                 channels,
                 reduction=2,
                 use_scale=True,
                 temperature=0.05):
        super(DisentangledSpatialSA, self).__init__()
        self.channels = channels
        self.inter_channels = max(channels // reduction, 1)
        self.use_scale = use_scale
        self.temperature = temperature
        self.qkv = nn.Conv2d(self.channels,
                             self.inter_channels * 3,
                             kernel_size=1)
        self.conv_out = nn.Conv2d(self.inter_channels,
                                  self.channels,
                                  kernel_size=1)
        self.init_weights()

    def init_weights(self, std=0.01):
        normal_init(self.qkv, std=std)
        constant_init(self.conv_out, 0)

    def forward(self, x):
        b, _, h, w = x.shape

        # q, k, v: B, C, H*W
        qkv = self.qkv(x) # B, C*3, H, W
        qkv = qkv.reshape(b, 3, self.inter_channels, h, w) # B, 3, C, H, W
        qkv = qkv.permute(1, 0, 2, 3, 4) # 3, B, C, H, W
        qkv = qkv.view(3, b, self.inter_channels, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # k, v: B, H*W, C
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # subtract mean
        k -= k.mean(dim=-2, keepdim=True)
        q -= q.mean(dim=-1, keepdim=True)

        # weight
        pairwise_weight = torch.matmul(k, q) # B, HxW, HxW
        if self.use_scale:
            pairwise_weight /= torch.tensor(
                self.inter_channels,
                dtype=torch.float,
                device=pairwise_weight.device) ** torch.tensor(
                0.5, device=pairwise_weight.device
            )
        pairwise_weight /= torch.tensor(
            self.temperature, device=pairwise_weight.device
        )
        pairwise_weight = pairwise_weight.softmax(dim=-1)

        y = torch.matmul(pairwise_weight, v) # B, HxW, C
        y = y.permute(0, 2, 1).contiguous().reshape(b, self.inter_channels,
                                                    h, w) # B, C, H, W

        out = x + self.conv_out(y)
        return out


class AttentionGate(nn.Module):
    def __init__(self,
                 in_channels: list,
                 out_channels,
                 return_weight=False):
        super(AttentionGate, self).__init__()
        assert out_channels == in_channels[1]
        self.return_weight = return_weight
        self.w_g = ConvModule(
            in_channels=in_channels[0],
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=None,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.w_x = ConvModule(
            in_channels=in_channels[1],
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=None,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.out_conv = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=dict(
                type='Sigmoid'
            ),
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.w_g(g)
        weight = self.w_x(x)
        weight = self.relu(g + weight)
        weight = self.out_conv(weight)

        return weight if self.return_weight else weight * x
