from turtle import forward
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .attention import SequentialPolarizedSelfAttention
from torch.nn import functional as F
from mmcv.cnn import ConvModule, xavier_init, constant_init

class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)
    
def decode(inp, inc, out, activation='swish', padding='same'):
    x = nn.ConvTranspose2d(inc, out, kernel_size=(4, 4), stride=(2, 2))(inp)
    x = nn.BatchNorm2d(out, momentum=0.9997, eps=4e-5)(x)
    x = nn.ReLU()(x)
    
    return x

class FEM(nn.Module):
    def __init__(self, in_channels=None, channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 interpolate_mode='bilinear', ops='cat', 
                 **kwargs):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        assert ops in ['cat', 'add']
        self.ops = ops
        self.in_channels = in_channels
        self.channels = channels

        self.e_convs = nn.ModuleList()
        for i in range(3):
            self.e_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels,
                              kernel_size=3, padding=(i*2+1), dilation=(i*2+1))
                )
            )
        self.fusion_conv = ConvModule(self.in_channels*3, self.channels,
                                      kernel_size=1, padding=0, norm_cfg=norm_cfg)
        

    def forward(self, input):
        inp_1 = self.e_convs[0](input)
        inp_2 = self.e_convs[1](input)
        inp_3 = self.e_convs[2](input)
        
        inp_ = torch.cat([inp_1, inp_2, inp_3], dim=1)
        
        out = self.fusion_conv(inp_)
        return out

class UpsampleAttention(nn.Module):
    def __init__(self, in_channels, mode="bilinear"):
        super().__init__()
        self.upsample_mode = mode
        self.spatial_weight = nn.Conv2d(
                in_channels + 1, 1, kernel_size=3, padding=1, bias=True)
        self.temp = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=True)
        for m in self.spatial_weight.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
    def _resize(self, x, size):
        if x.shape[-2:] == size:
            return x
        elif x.shape[-2:] != size:
            return F.interpolate(x, size=size, mode=self.upsample_mode)

            return x
    def forward(self, x, x_up):
        x1 = x
        x2 = self._resize(x_up, size=x1.shape[-2:])
        batch, channel, height, width = x1.size()
        
        upsample_weight = (
            self.temp * channel**(-0.5) *
            self.spatial_weight(torch.cat((x1, x2), dim=1)))
        upsample_weight = F.softmax(
            upsample_weight.reshape(batch, 1, -1), dim=-1).reshape(
            batch, 1, height, width) * height * width
        x2 = upsample_weight * x2
        return x2
                
                
class Uncertain_Boundary(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv0 = ConvBlock(in_channel, 1, kernel_size=1, padding=0)
        self.conv1 = ConvBlock(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(in_channel, out_channel, kernel_size=3, padding=1)
        self.gate = SequentialPolarizedSelfAttention(in_channel)
        self.l_temp = torch.nn.Parameter(0.25, requires_grad=True)
        self.h_temp = torch.nn.Parameter(0.75, requires_grad=True)

    def forward(self, feature):
        mask = self.conv0(feature)
        s_mask = torch.sigmoid(mask)
        t_mask = s_mask.clone()
        t_mask[t_mask<self.l_temp] = -1
        t_mask[(t_mask>=self.l_temp) & (t_mask<=self.h_temp)] = 1
        t_mask[t_mask>self.h_temp] = 0
        
        feature_ = F.relu(feature)
        feature_ = t_mask.expand_as(feature_).mul(feature_)
        feature_ = self.gate(self.conv1(feature_))
        feature_ = self.conv2(feature_ + s_mask)
        return feature_