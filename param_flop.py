# model segformerb4 settings
from mmseg.models import build_segmentor
from mmcv.runner.optimizer import build_optimizer
import torch

def segformer(arch):
    num_layers = []
    pretrained = f'pretrained/mit_{arch}_mmseg.pth'
    if arch == 'b1':
        num_layers = [2,2,2,2]
    if arch == 'b2':
        num_layers = [3, 4, 6, 3 ]
    if arch == 'b3':
        num_layers = [3, 4, 18, 3]
    if arch == 'b4':
        num_layers = [3, 8, 27, 3]
    model = dict(
        type='SunSegmentor',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=num_layers,
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            pretrained=pretrained),
        decode_head=dict(
            type='DRPHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'))
    model = build_segmentor(model)
    model.init_weights()
    return model
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

from mmcv.cnn import get_model_complexity_info

def get_model_info(model, tsize):
    model.eval()

    input_shape = (3, tsize[0], tsize[1])
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    
model = segformer('b2')

get_model_info(model, (352, 352))

'''
B1: Flops: 7.21 GFLOPs
Params: 17.46 M

B2: Flops: 10.18 GFLOPs
Params: 28.51 M
[Flops: 11.33 GFLOPs
Params: 29.29 M]

B3: Flops: 15.63 GFLOPs
Params: 48.38 M
[Flops: 16.78 GFLOPs
Params: 49.17 M]


B4: Flops: 20.9 GFLOPs
Params: 65.15 M
'''