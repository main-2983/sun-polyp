from mcode.utils import get_model_info
from mmseg.models.builder import build_segmentor

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


model_cfg1 = dict(
    type='SunSegmentor',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=None),
    decode_head=dict(
        type='MLP_OSAHead_v5',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)
model_cfg2 = dict(
    type='SunSegmentor',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=None),
    decode_head=dict(
        type='SSFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

model1 = build_segmentor(model_cfg1)
model2 = build_segmentor(model_cfg2)

get_model_info(model1, (352, 352))
get_model_info(model2, (352, 352))

if __name__ == '__main__':
    image_path = "/home/nguyen.mai/Desktop/299877739_797830788010518_2585821702899177673_n.jpg"
    image_c = cv2.imread(image_path)[:, :, ::-1]
    print(image_c.shape)
    image_p = np.array(Image.open(image_path))
    print(image_p.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(image_c[:20, :20, 0])
    print(image_c[:20][:20] - image_p[:20][:20])
    plt.subplot(1, 2, 2)
    plt.imshow(image_p[:20, :20, 0])
    plt.show()