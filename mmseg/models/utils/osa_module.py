import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


__all__ = [
    'OSAConv', 'OSAConvV2', 'OSAConvV3'
]


class OSAConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 kernel_size=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(OSAConv, self).__init__()
        mid_channels = out_channels // num_convs
        self.pwconv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

    def forward(self, x):
        out = self.pwconv(x)
        outs = []
        for i in range(len(self.convs)):
            _out = self.convs[i](out)
            outs.append(_out)
            out = _out
        out = torch.cat(outs, dim=1)
        return out


class OSAConvV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 kernel_size=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(OSAConvV2, self).__init__()
        mid_channels = out_channels // num_convs
        self.pwconv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
        self.pwconv2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=None,
            norm_cfg=None
        )

    def forward(self, x):
        out = self.pwconv(x)
        outs = []
        for i in range(len(self.convs)):
            _out = self.convs[i](out)
            outs.append(_out)
            out = _out
        out = torch.cat(outs, dim=1)
        out = self.pwconv2(out)
        return out


class OSAConvV3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 kernel_size=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU'
                 )):
        super(OSAConvV3, self).__init__()
        mid_channels = out_channels // num_convs
        self.pwconv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
        self.pwconv2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        out = self.pwconv(x)
        outs = []
        for i in range(len(self.convs)):
            _out = self.convs[i](out)
            outs.append(_out)
            out = _out
        out = torch.cat(outs, dim=1)
        out = self.pwconv2(out)
        return out


class ELANConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=4,
                 kernel_size=3,
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 act_cfg=dict(
                     type='ReLU',
                 )):
        super(ELANConv, self).__init__()
        assert num_convs % 2 == 0
        num_groups = num_convs // 2
        mid_channels = out_channels // num_groups

        self.pwconv = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.convs = nn.ModuleList()
        for i in range(num_groups):
            self.convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size//2,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg
                    ),
                    ConvModule(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size//2,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg
                    )
                )
            )

    def forward(self, x):
        out = self.pwconv(x)
        outs = []
        for i in range(len(self.convs)):
            _out = self.convs[i](out)
            outs.append(_out)
            out = _out