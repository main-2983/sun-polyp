import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

from mmseg.models.builder import build_segmentor

from mcode import select_device, UnNormalize


# config
ckpt_path = "logs/MLPOSA_v5/model_50.pth"
image_path = "../Dataset/polyp/TestDataset/CVC-300/images/150.png"
mask_path = "../Dataset/polyp/TestDataset/CVC-300/masks/150.png"
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
        type='MLP_OSAHead_v5',
        ops='cat',
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
    # init
    device = select_device('')

    # model
    model = build_segmentor(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (352, 352))[:, :, ::-1]
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (352, 352))[:, :, 0]
    sample = transform(image=image, mask=mask)
    img, gt_mask = sample["image"], sample["mask"]
    gt_mask = np.asarray(gt_mask, np.float32)
    img = img[None].to(device)

    with torch.no_grad():
        feat_maps = model(img) # this will return feature maps of choice set explicitly by returning in decode_head
        feat_maps = feat_maps.cpu().numpy()
    _, c, _, _ = feat_maps.shape
    print(f"Feature maps shape: {feat_maps.shape}")
    nrows, ncols = int(np.sqrt(c)), int(np.sqrt(c))

    fig_size = nrows * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size, fig_size))
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].imshow(feat_maps[0, i + j, :, :])
            axes[i, j].axis('off')
    plt.savefig("test5.jpg")