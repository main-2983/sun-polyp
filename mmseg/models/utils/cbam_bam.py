import torch.nn as nn

from mmcv.cnn import ConvModule


class BAMSpatial(nn.Module):
    def __init__(self,
                 channels,
                 reduction=16,
                 dilation=2):
        super(BAMSpatial, self).__init__()
        self.spatial = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels//reduction,
                kernel_size=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            ConvModule(
                in_channels=channels//reduction,
                out_channels=channels//reduction,
                kernel_size=3,
                padding=2,
                dilation=dilation,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            ConvModule(
                in_channels=channels//reduction,
                out_channels=channels//reduction,
                kernel_size=3,
                padding=2,
                dilation=dilation,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            )
        )
        self.squeeze = nn.Conv2d(
            in_channels=channels//reduction,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        _, c, _, _ = x.shape
        weight = self.spatial(x)
        weight = self.squeeze(weight)
        weight = weight.expand(-1, c, -1, -1)

        return weight * x