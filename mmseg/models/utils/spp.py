from mmcv.cnn import ConvModule

import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_sizes=(5, 9, 13)):
        ConvModule


class CSPSPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio=0.5,
                 kernel_sizes=(5, 9, 13)):
        super(CSPSPP, self).__init__()
        _c = int(in_channels * ratio)
        self.split1 = ConvModule(
            in_channels=in_channels,
            out_channels=_c,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.split2 = nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=_c,
                kernel_size=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            nn.Sequential(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=_c,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(
                        type='BN',
                        requires_grad=True
                    )
                ),
                ConvModule(
                    in_channels=in_channels,
                    out_channels=_c,
                    kernel_size=1,
                    norm_cfg=dict(
                        type='BN',
                        requires_grad=True
                    )
                )
            )
        ])