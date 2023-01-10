from ..builder import HEADS
import torch
from .decode_head import BaseDecodeHead
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
from icecream import ic 
from .lib.psa import PSA_p
from .psp_head import PPM

######################################################################################################################

# pre-activation based upsampling conv block
class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, act, norm="BN", num_groups=1):
        super(upConvLayer, self).__init__()
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.drop = nn.Dropout(0.1)
        self.act = act
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)     #pre-activation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        x = self.drop(x)
        return x

def make_padding(kernel_size, dilation):
    if type(kernel_size) != int:
        index = kernel_size.index(max(kernel_size))
        kernel_size = max(kernel_size)
        width_pad_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        if index == 0:
            return (width_pad_size, 0)
        else:
            return (0, width_pad_size)
    else:
        width_pad_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
    return width_pad_size

# pre-activation based conv block
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='BN', act='ReLU', num_groups=1):
        super(Conv, self).__init__()
        padding = make_padding(kSize, dilation)
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(nn.Conv2d(in_ch, out_ch, kernel_size=kSize, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=num_groups))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

class AttentionModule(nn.Module):
    def __init__(self, infeat):
        super().__init__()
        dim = infeat//2
        self.reduc_conv = Conv(infeat, dim, 1)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.reduc_conv(x)
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

# ASPP Module
class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        # 1 : large separable depthwise conv 

        self.reduction1 = Conv(in_feat, in_feat//2, kSize=1, stride = 1, bias=False, padding=0)
        dim = in_feat//2
        self.aspp_d3 = nn.Sequential(Conv(dim, dim, kSize=11, stride=1, dilation=1,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        self.attn_reduc1 = Conv(in_feat, dim, kSize=1)
        self.aspp_d6 = nn.Sequential(Conv(dim, dim, kSize=11, stride=1, dilation=2,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        self.attn_reduc2 = Conv(in_feat, dim, kSize=1)
        self.aspp_d9 = nn.Sequential(Conv(dim, dim, kSize=11, stride=1, dilation=4,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=1, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        
        self.reduction2 = Conv((dim)*4, dim, kSize=1, stride=1, padding=0,bias=False, norm=norm, act=act)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = self.attn_reduc1(torch.cat([x, d3],dim=1))
        d6 = self.aspp_d6(cat1)
        cat2 = self.attn_reduc2(torch.cat([cat1, d6],dim=1))
        d9 = self.aspp_d9(cat2)
        out = self.reduction2(torch.cat([x,d3,d6,d9], dim=1))
        return out      # 256 x H/16 x W/16


# ASPP Module
class Dilated_bottleNeck_1(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck_1, self).__init__()
        # less channel large conv
        self.reduction1 = Conv(in_feat, in_feat//2, kSize=1, stride = 1, bias=False, padding=0)

        self.aspp_d3 = nn.Sequential(Conv(in_feat//2, in_feat//4, kSize=1, stride=1, dilation=1,bias=False, norm=norm, act=act),
                                     Conv(in_feat//4, in_feat//4, kSize=9, stride=1, padding=0, dilation=1,bias=False, norm=norm, act=act))
        self.attn_reduc1 = Conv(in_feat//2 + in_feat//4, in_feat//2, kSize=1)
        self.aspp_d6 = nn.Sequential(Conv(in_feat//2, in_feat//4, kSize=1, stride=1, dilation=1,bias=False, norm=norm, act=act),
                                     Conv(in_feat//4, in_feat//4, kSize=9, stride=1, padding=0, dilation=2,bias=False, norm=norm, act=act))
        self.attn_reduc2 = Conv(in_feat//2 + in_feat//4, in_feat//2, kSize=1)
        self.aspp_d9 = nn.Sequential(Conv(in_feat//2, in_feat//4, kSize=1, stride=1, dilation=4,bias=False, norm=norm, act=act),
                                     Conv(in_feat//4, in_feat//4, kSize=9, stride=1, padding=0, dilation=4,bias=False, norm=norm, act=act))
        
        self.reduction2 = Conv((in_feat//4)*3 + in_feat//2, in_feat//2, kSize=1, stride=1, padding=0,bias=False, norm=norm, act=act)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = self.attn_reduc1(torch.cat([x, d3],dim=1))
        d6 = self.aspp_d6(cat1)
        cat2 = self.attn_reduc2(torch.cat([cat1, d6],dim=1))
        d9 = self.aspp_d9(cat2)
        out = self.reduction2(torch.cat([x,d3,d6,d9], dim=1))
        return out      # 256 x H/16 x W/16


# ASPP Module
class Dilated_bottleNeck_2(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck_2, self).__init__()

        self.reduction1 = Conv(in_feat, in_feat//2, kSize=1, stride = 1, bias=False, padding=0)
        dim = in_feat//2
        self.aspp_d3 = nn.Sequential(Conv(dim, dim, kSize=(9,1), stride=1,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=(1,9), stride=1,bias=False, norm=norm, act=act, num_groups=dim))
        self.attn_reduc1 = Conv(in_feat, dim, kSize=1)
        self.aspp_d6 = nn.Sequential(Conv(dim, dim, kSize=(11,1), stride=1,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=(1,11), stride=1,bias=False, norm=norm, act=act, num_groups=dim))
        self.attn_reduc2 = Conv(in_feat, dim, kSize=1)
        self.aspp_d9 = nn.Sequential(Conv(dim, dim, kSize=(13,1), stride=1,bias=False, norm=norm, act=act, num_groups=dim),
                                     Conv(dim, dim, kSize=(1,13), stride=1,bias=False, norm=norm, act=act, num_groups=dim))
        
        self.reduction2 = Conv((dim)*4, dim, kSize=1, stride=1,bias=False, norm=norm, act=act)
    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = self.attn_reduc1(torch.cat([x, d3],dim=1))
        d6 = self.aspp_d6(cat1)
        cat2 = self.attn_reduc2(torch.cat([cat1, d6],dim=1))
        d9 = self.aspp_d9(cat2)
        out = self.reduction2(torch.cat([x,d3,d6,d9], dim=1))
        return out      # 256 x H/16 x W/16



# 1 : large separable depthwise conv - Params:0.954 Flops:0.116 - 87.2: kvasir_93.13, colon_80.8, etis_82.3, cliniic_92.1
# *2 : less channels large conv  - Params:4.578 Flops:0.554 - 86.8: kvasir_92.89, colon_81.7, etis_79.8
# 3 : strip conv                 - Params:4.989 Flops:0.604 - 86.9: kvasir_92.09, colon_81.6, etis_81.1
# 4 : strip depthwise conv       - Params:0.681 Flops:0.083 - 87.3: kvasir_92.89, colon_81.4, etis_82
# 5 : large conv attn module       Params:0.223 Flops:0.027 - 87.38: kvasir_92.22, colon_81.9, etis_81.4
# DEEP RESIDUAL PYRAMID HEAD
@HEADS.register_module()
class DRPHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        norm = "BN"
        act = 'ReLU'
        kSize = 3
        ############################################     Pyramid Level 5     ###################################################
        # decoder1 out : 1 x H/16 x W/16 (Level 5)
        # self.ASPP = Dilated_bottleNeck(norm, act, self.in_channels[3])
        self.ASPP = AttentionModule(self.in_channels[3]) 
        
        self.psa_1 = PSA_p(self.in_channels[3]//2, self.in_channels[3]//2) 
        self.decoder1_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.decoder1 = nn.Sequential(Conv(self.in_channels[3]//2, self.in_channels[3]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),      
                                        Conv(self.in_channels[3]//4, self.in_channels[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),    
                                        Conv(self.in_channels[3]//8, self.in_channels[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),  
                                        Conv(self.in_channels[3]//16, self.in_channels[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act),
                                        Conv(self.in_channels[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                            norm=norm, act=act)
                                     )
        
        ########################################################################################################################

        ############################################     Pyramid Level 4     ###################################################
        # decoder2 out : 1 x H/8 x W/8 (Level 4)
        # decoder2_up : (H/16,W/16)->(H/8,W/8)
        
        self.decoder2_up1 = upConvLayer(self.in_channels[3]//2, self.in_channels[3]//2, 2, norm, act)
        self.decoder2_attn = PSA_p(self.in_channels[3]//2, self.in_channels[3]//2)
        self.decoder2_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.decoder2_reduc1 = Conv(self.in_channels[3]//2 + self.in_channels[2], self.in_channels[3]//2, kSize=kSize, stride=1, padding=kSize//2,bias=False, 
                                        norm=norm, act=act)
        self.decoder2_1 = Conv(self.in_channels[3]//2, self.in_channels[3]//4, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder2_2 = Conv(self.in_channels[3]//4, self.in_channels[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder2_3 = Conv(self.in_channels[3]//8, self.in_channels[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder2_4 = Conv(self.in_channels[3]//16, self.in_channels[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder2_5 = Conv(self.in_channels[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        
        ########################################################################################################################

        ############################################     Pyramid Level 3     ###################################################
        # decoder3 out : 1 x H/4 x W/4 (Level 3)
        # decoder3_up : (H/8,W/8)->(H/4,W/4)
        
        self.decoder3_up2 = upConvLayer(self.in_channels[3]//4, self.in_channels[3]//4, 2, norm, act, (self.in_channels[3]//4)//16)
        self.decoder3_attn = PSA_p(self.in_channels[3]//4, self.in_channels[3]//4)
        self.decoder3_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.decoder3_reduc2 = Conv(self.in_channels[3]//4 + self.in_channels[1], self.in_channels[3]//4, kSize=kSize, stride=1, padding=kSize//2,bias=False, 
                                        norm=norm, act=act)
        self.decoder3_1 = Conv(self.in_channels[3]//4, self.in_channels[3]//8, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        
        self.decoder3_2 = Conv(self.in_channels[3]//8, self.in_channels[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        
        self.decoder3_3 = Conv(self.in_channels[3]//16, self.in_channels[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder3_4 = Conv(self.in_channels[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        
        ########################################################################################################################

        ############################################     Pyramid Level 2     ###################################################
        # decoder4 out : 1 x H/2 x W/2 (Level 2)
        # decoder4_up : (H/4,W/4)->(H/2,W/2)
        
        self.decoder4_up3 = upConvLayer(self.in_channels[3]//8, self.in_channels[3]//8, 2, norm, act, (self.in_channels[3]//8)//16)
        self.decoder4_attn = PSA_p(self.in_channels[3]//8, self.in_channels[3]//8)
        self.decoder4_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.decoder4_reduc3 = Conv(self.in_channels[3]//8 + self.in_channels[0], self.in_channels[3]//8, kSize=kSize, stride=1, padding=kSize//2,bias=False, 
                                        norm=norm, act=act)
        self.decoder4_1 = Conv(self.in_channels[3]//8, self.in_channels[3]//16, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder4_2 = Conv(self.in_channels[3]//16, self.in_channels[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder4_3 = Conv(self.in_channels[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        
        ########################################################################################################################
        
        ############################################     Pyramid Level 1     ###################################################
        # decoder5 out : 1 x H x W (Level 1)
        # decoder5_up : (H/2,W/2)->(H,W)
        
        self.decoder5_up4 = upConvLayer(self.in_channels[3]//16, self.in_channels[3]//16, 2, norm, act, (self.in_channels[3]//16)//16)
        self.decoder5_attn = PSA_p(self.in_channels[3]//16, self.in_channels[3]//16)
        self.decoder5_temp = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.decoder5_reduc4 = Conv(self.in_channels[3]//16, self.in_channels[3]//16, kSize=kSize, stride=1, padding=kSize//2,bias=False, 
                                        norm=norm, act=act)
        self.decoder5_1 = Conv(self.in_channels[3]//16, self.in_channels[3]//32, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        self.decoder5_2 = Conv(self.in_channels[3]//32, 1, kSize, stride=1, padding=kSize//2, bias=False, 
                                        norm=norm, act=act)
        ########################################################################################################################
        
        self.upscale = F.interpolate

    def forward(self, inputs):
        inputs, laplacian = inputs
        x = self._transform_inputs(inputs)
        cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3]
        rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = laplacian

        
        # decoder 1 - Pyramid level 5
        dense_feat = self.ASPP(dense_feat)   
        dense_feat = self.psa_1(dense_feat)                # Dense feature for lev 5
        self.mask_lv5 = self.decoder1(dense_feat)# + (self.decoder1_temp*rgb_lv5.mean(dim=1,keepdim=True))  # block 1-1  -  R5
        mask_lv5_up = self.upscale(self.mask_lv5, scale_factor = 2, mode='bilinear')


        # decoder 2 - Pyramid level 4
        dec2 = self.decoder2_up1(dense_feat)    # Upconv 1  
        dec2 = self.decoder2_reduc1(torch.cat([dec2,cat3],dim=1))    # X4
        dec2 = self.decoder2_attn(torch.sigmoid(mask_lv5_up)*dec2)
        dec2_up = self.decoder2_1(dec2)     #  block 2-1
        dec2 = self.decoder2_2(dec2_up)     #  block 2-2
        dec2 = self.decoder2_3(dec2)        #  block 2-3
        dec2 = self.decoder2_4(dec2)        #  block 2-4
        self.mask_lv4 = self.decoder2_5(dec2)# + (self.decoder2_temp*rgb_lv4.mean(dim=1,keepdim=True))    #  block 2-5  -  R4 
        mask_lv4_up = self.upscale(self.mask_lv4, scale_factor = 2, mode='bilinear')
        
        
        # decoder 2 - Pyramid level 3
        dec3 = self.decoder3_up2(dec2_up)     # Upconv 2
        dec3 = self.decoder3_reduc2(torch.cat([dec3,cat2],dim=1))    # X3
        dec3 = self.decoder3_attn(torch.sigmoid(mask_lv4_up)*dec3)
        dec3_up = self.decoder3_1(dec3)     #  block 3-1
        dec3 = self.decoder3_2(dec3_up)     #  block 3-2
        dec3 = self.decoder3_3(dec3)        #  block 3-3
        self.mask_lv3 = self.decoder3_4(dec3)# + (self.decoder3_temp*rgb_lv3.mean(dim=1,keepdim=True))     #  block 3-3     R3
        mask_lv3_up = self.upscale(self.mask_lv3, scale_factor = 2, mode='bilinear')
        
        
        # decoder 2 - Pyramid level 2
        dec4 = self.decoder4_up3(dec3_up)   # Upconv 3
        dec4 = self.decoder4_reduc3(torch.cat([dec4,cat1],dim=1))   # X2
        dec4 = self.decoder4_attn(torch.sigmoid(mask_lv3_up)*dec4)
        dec4_up = self.decoder4_1(dec4)   #  block 4-1
        dec4 = self.decoder4_2(dec4_up)   #  block 4-2     R2
        self.mask_lv2 = self.decoder4_3(dec4)# + (self.decoder4_temp*rgb_lv2.mean(dim=1,keepdim=True))  #  block 4-3     R2
        mask_lv2_up = self.upscale(self.mask_lv2, scale_factor = 2, mode='bilinear')
        
        
        # decoder 2 - Pyramid level 1
        dec5 = self.decoder5_up4(dec4_up) # Upconv 4
        dec5 = self.decoder5_attn(torch.sigmoid(mask_lv2_up)*dec5)
        dec5 = self.decoder5_1(dec5)     #  block 5-1
        self.mask_lv1 = self.decoder5_2(dec5)# + (self.decoder5_temp*rgb_lv1.mean(dim=1,keepdim=True))    #  block 5-2
        
        
        # mask restoration
        mask_lv4_img = self.mask_lv4 + mask_lv5_up
        mask_lv3_img = self.mask_lv3 + self.upscale(mask_lv4_img, scale_factor = 2, mode = 'bilinear')
        mask_lv2_img = self.mask_lv2 + self.upscale(mask_lv3_img, scale_factor = 2, mode = 'bilinear')
        final_mask = self.mask_lv1 + self.upscale(mask_lv2_img, scale_factor = 2, mode = 'bilinear')

        return final_mask