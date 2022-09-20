from mmcv.cnn import ConvModule

import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k,
                         stride=1,
                         padding=k // 2) for k in kernel_sizes
        ])
        self.out_conv = ConvModule(
            in_channels=in_channels * (len(kernel_sizes) + 1),
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )

    def forward(self, x):
        x = self.conv(x)
        feats = torch.cat([x] + [pooling(x) for pooling in self.pooling], dim=1)
        out = self.out_conv(feats)
        return out


class CSPSPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 ratio=0.5,
                 kernel_sizes=(5, 9, 13)):
        super(CSPSPP, self).__init__()
        if out_channels is None:
            out_channels = in_channels
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
        self.split2 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=_c,
                kernel_size=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            ConvModule(
                in_channels=_c,
                out_channels=_c,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            SPP(
                in_channels=_c,
                kernel_sizes=kernel_sizes
            ),
            ConvModule(
                in_channels=_c,
                out_channels=_c,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            )
        )
        self.out_conv = ConvModule(
            in_channels=_c * 2,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )

    def forward(self, x):
        feat_split1 = self.split1(x)
        feat_split2 = self.split2(x)
        out = self.out_conv(torch.cat([feat_split1, feat_split2], dim=1))
        return out


class SPPF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=5):
        super(SPPF, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size // 2)
        self.out_conv = ConvModule(
            in_channels=in_channels * 4,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )

    def forward(self, x):
        x = self.conv(x)
        x1 = self.pooling(x)
        x2 = self.pooling(x1)
        x3 = self.pooling(x2)
        feats = torch.cat([x, x1, x2, x3], dim=1)
        out = self.out_conv(feats)
        return out


class CSPSPPF(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 ratio=0.5,
                 kernel_size=5):
        super(CSPSPPF, self).__init__()
        if out_channels is None:
            out_channels = in_channels
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
        self.split2 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=_c,
                kernel_size=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            ConvModule(
                in_channels=_c,
                out_channels=_c,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            ),
            SPPF(
                in_channels=_c,
                kernel_size=kernel_size
            ),
            ConvModule(
                in_channels=_c,
                out_channels=_c,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(
                    type='BN',
                    requires_grad=True
                )
            )
        )
        self.out_conv = ConvModule(
            in_channels=_c * 2,
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=dict(
                type='BN',
                requires_grad=True
            )
        )

    def forward(self, x):
        feat_split1 = self.split1(x)
        feat_split2 = self.split2(x)
        out = self.out_conv(torch.cat([feat_split1, feat_split2], dim=1))
        return out
