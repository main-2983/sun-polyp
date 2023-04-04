import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

from mmseg.models.builder import build_segmentor

from mcode import select_device, UnNormalize
from torch_hooks import IOHook


# config
AVG = True
ckpt_path = "ckpts/exfuse-add-scale-dropC-se(2).pth"
image_path = "Dataset/TestDataset/CVC-300/images/206.png"
mask_path = "Dataset/TestDataset/CVC-300/masks/206.png"
# image_path = "Dataset/TrainDataset/image/206.png"
# mask_path = "Dataset/TrainDataset/mask/206.png"
save_path = "runs/featmap/v2(11)"
target_params = [
    "model.decode_head.scales[0].scale",
    # "model.decode_head.scales[1].scale",
    # "model.decode_head.scales[2].scale",
    "model.decode_head.scales[3].scale",
]
# target_layer = None
input = True
num_chans = None
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
model_cfg = dict(
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
        type='LAPHead_v2_35',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)


if __name__ == '__main__':
    # model
    model = build_segmentor(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    values = [eval(target_param) for target_param in target_params]
    values = [value.detach().numpy().squeeze() for value in values]
    for idx, value in enumerate(values):
        plt.plot(value)
        plt.legend(target_params)
    plt.show()



