import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

class PartialDecoder(nn.Module):
    """ This implementation is a little different from PraNet
    in PraNet, the last Conv layer is actually a layer to generate prediction,
    we remove it in this implementation and move the prediction layer to decode head
    """
    def __init__(self, channels):
        super(PartialDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample3 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample4 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_upsample5 = ConvModule(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            act_cfg=None)

        self.conv_cat2 = ConvModule(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            act_cfg=None)
        self.conv_cat3 = ConvModule(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            act_cfg=None)

        self.conv4 = ConvModule(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            act_cfg=None)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_cat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_cat3(x3_2)

        x = self.conv4(x3_2)

        return x