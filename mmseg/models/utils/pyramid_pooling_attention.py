import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, build_activation_layer

from mmseg.ops import resize


class PyramidPoolingAttention(nn.Module):
    """
    This module use PPM (Pyramid Pooling Module) as an Attention module
    The attention weight is the combination of different pool scale
    """
    def __init__(self,
                 channels,
                 act_cfg=None,
                 pool_scales=(1, 2, 3, 6)):
        super(PyramidPoolingAttention, self).__init__()
        self.pools = nn.ModuleList()
        for pool_scale in pool_scales:
            self.pools.append(
                nn.AdaptiveAvgPool2d(pool_scale)
            )
        self.bottleneck = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            act_cfg=act_cfg
        )

    def forward(self, x):
        ppm_outs = []
        for pool in self.pools:
            out = pool(x)
            upsampled_out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            ppm_outs.append(upsampled_out)
        attn_weight = sum(ppm_outs)
        attn_weight = self.bottleneck(attn_weight)
        return x * attn_weight


class PyramidPoolingAttentionv2(nn.Module):
    """
    This module use PPM (Pyramid Pooling Module) as an Attention module
    The attention weight is the combination of different pool scale
    The difference between this and v1 is: v2 doesn't include a bottleneck
    """
    def __init__(self,
                 channels,
                 act_cfg=None,
                 pool_scales=(1, 2, 3, 6)):
        super(PyramidPoolingAttentionv2, self).__init__()
        self.pools = nn.ModuleList()
        for pool_scale in pool_scales:
            self.pools.append(
                nn.AdaptiveAvgPool2d(pool_scale)
            )
        self.bottleneck = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            act_cfg=act_cfg
        )

    def forward(self, x):
        ppm_outs = [x]
        for pool in self.pools:
            out = pool(x)
            upsampled_out = resize(
                input=out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            ppm_outs.append(upsampled_out)
        attn_weight = sum(ppm_outs)
        attn_weight = self.bottleneck(attn_weight)
        return x * attn_weight


class PyramidPoolingSelfAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_scales=(1, 2, 3, 6),
                 act='softmax'):
        assert act in ['softmax', 'sigmoid'], \
            f"Activation for {self.__class__} must be 'softmax' or 'sigmoid'"
        super(PyramidPoolingSelfAttention, self).__init__()
        self.pools = nn.ModuleList()
        for pool_scale in pool_scales:
            self.pools.append(
                nn.AdaptiveAvgPool2d(pool_scale)
            )
        self.out_channels = out_channels
        self.kv = nn.Linear(
            in_features=in_channels,
            out_features=out_channels * 2
        )
        self.q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        if act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        else:
            self.act = nn.Sigmoid()

    def forward_pp(self, x):
        pp_outs = []
        for pool in self.pools:
            out = pool(x) # B, C, H, W
            out = out.reshape(*out.shape[:2], -1) # B, C, H*W
            out = out.permute(0, 2, 1).contiguous() # B, H*W, C
            pp_outs.append(out)
        pp_outs = torch.cat(pp_outs, dim=1)
        return pp_outs

    def forward(self, x):
        B, _, H, W = x.shape

        pp_out = self.forward_pp(x)
        kv = self.kv(pp_out) # B, 50, C*2
        kv = kv.reshape(B, -1, 2, self.out_channels) # B, 50, 2, C
        kv = kv.permute(2, 0, 1, 3) # 2, B, 50, C
        k, v = kv[0], kv[1] # B, 50, C each
        q = self.q(x) # B, C, H, W
        q = q.reshape(*q.shape[:2], -1).permute(0, 2, 1) # B, L, C
        attn = q@k.transpose(1, 2) # B, L, 50
        attn = self.act(attn)
        out = attn@v # B, L, C
        out = out.permute(0, 2, 1).contiguous() # B, C, L
        out = out.reshape(B, self.out_channels, H, W)

        return out
