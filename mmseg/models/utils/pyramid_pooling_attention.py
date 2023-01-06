import torch.nn as nn

from mmcv.cnn import ConvModule

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