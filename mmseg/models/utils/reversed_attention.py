import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize

class ReversedAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 num_convs=2):
        super(ReversedAttention, self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        # Conv + BN
        self.bottleneck = ConvModule(in_channels,
                                     channels,
                                     kernel_size=1,
                                     norm_cfg=norm_cfg,
                                     act_cfg=None)
        convs = []
        # Conv + BN + ReLU
        for i in range(num_convs):
            convs.append(
                ConvModule(channels,
                           channels,
                           kernel_size=kernel_size,
                           padding=kernel_size//2,
                           norm_cfg=norm_cfg)
            )
        self.convs = nn.Sequential(*convs)
        self.pred_conv = ConvModule(channels,
                                    out_channels=1,
                                    kernel_size=1,
                                    norm_cfg=norm_cfg,
                                    act_cfg=None)

    def forward(self, x, y):
        """
        Forward function of Reversed Attention
        Args:
            y: prediction mask (B, 1, H, W)
            x: feature maps
        """
        assert y.shape[1] == 1, "'y' must be a prediction mask"
        _, c, h, w = x.shape

        y_res = resize(input=y,
                       size=(h, w),
                       mode='bilinear')
        rev_y = -1 * torch.sigmoid(y_res) + 1
        rev_y = rev_y.expand(-1, c, -1, -1)
        x = rev_y * x
        x = self.bottleneck(x)
        x = self.convs(x)
        x_pred = self.pred_conv(x)
        out = x_pred + y_res
        return out
