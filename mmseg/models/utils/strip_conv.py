import torch
import torch.nn as nn

from mmcv.cnn import ConvModule


__all__ = [
    'SeqStripConv', 'ParStripConv',
    'MultiScaleStripConv'
]


class SeqStripConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 act_cfg=dict(
                     type='ReLU'
                 ),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 )):
        super(SeqStripConv, self).__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size//2),
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.conv2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.pw = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pw(out)
        return out


class ParStripConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 act_cfg=dict(
                     type='ReLU'
                 ),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 )):
        super(ParStripConv, self).__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size//2),
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.conv2 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size//2, 0),
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.pw = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = out1 + out2
        out = self.pw(out)

        return out


class MultiScaleStripConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(3, 5),
                 act_cfg=dict(
                     type='ReLU'
                 ),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 ),
                 ops='add'):
        super(MultiScaleStripConv, self).__init__()
        assert ops in ['add', 'cat']
        self.ops = ops
        self.strip_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            conv1 = ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                act_cfg=act_cfg,
                norm_cfg=norm_cfg
            )
            conv2 = ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                act_cfg=act_cfg,
                norm_cfg=norm_cfg
            )
            self.strip_convs.append(
                nn.Sequential(conv1, conv2)
            )
        self.pw = ConvModule(
            in_channels=in_channels if ops == 'add' else in_channels*len(kernel_sizes),
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        outs = []
        for strip_conv in self.strip_convs:
            outs.append(strip_conv(x))
        if self.ops == 'add':
            outs = sum(outs)
        else:
            outs = torch.cat(outs, dim=1)
        out = self.pw(outs)

        return out