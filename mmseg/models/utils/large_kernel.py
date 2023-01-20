import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


__all__ = [
    'LargeKernelDWConv'
]


# This scripts contains multiple Large Kernel Conv/Attention impelementation
# Such as:
# - a normal Large Kernel Conv
# - a combo of Large Kernel Conv like LKA in VAN
# - and its attention variation


# a normal Large Kernel Conv consists of a DW Large Kernel Conv and a pw Conv
class LargeKernelDWConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 act_cfg=dict(
                     type='ReLU'
                 ),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 )):
        super(LargeKernelDWConv, self).__init__()
        self.lk_conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            groups=in_channels
        )
        self.pw_conv = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        out = self.lk_conv(x)
        out = self.pw_conv(out)

        return out
