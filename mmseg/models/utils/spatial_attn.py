import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


__all__ = [
    'AvgSpatialAttn'
]


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
                     type='Sigmoid'
                 )):
        super(AvgSpatialAttn, self).__init__()
        self.conv = ConvModule(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.channels = channels

    def forward(self, x):
        weight:torch.Tensor = torch.mean(x, dim=1, keepdim=True)
        weight = weight.expand(-1, self.channels, -1, -1)
        return weight * x
