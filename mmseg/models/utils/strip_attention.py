import torch.nn as nn

from mmcv.cnn import ConvModule, build_activation_layer


class StripPoolingAttention(nn.Module):
    """
    References: https://arxiv.org/pdf/2003.13328.pdf
    """
    def __init__(self,
                 channels,
                 act_cfg=dict(
                     type='Sigmoid'
                 ),
                 mid_act=dict(
                     type='ReLU'
                 ),
                 norm_cfg=dict(
                     type='BN',
                     requires_grad=True
                 )):
        super(StripPoolingAttention, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            norm_cfg=norm_cfg,
            act_cfg=None
        )
        self.conv2 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            norm_cfg=norm_cfg,
            act_cfg=None
        )

        self.act = nn.Identity()
        if mid_act is not None:
            self.act = build_activation_layer(mid_act)

        self.conv3 = ConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            norm_cfg=None,
            act_cfg=act_cfg
        )

    def forward(self, x):
        _, _, h, w = x.shape
        # 1st branch
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = x1.expand(-1, -1, h, w)

        # 2nd branch
        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = x2.expand(-1, -1, h, w)

        weight = self.act(x1 + x2)
        weight = self.conv3(weight)

        return x * weight


