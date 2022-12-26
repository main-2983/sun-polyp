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
AVG = False
ckpt_path = "ckpts/LAPFormer-B1.pth"
image_path = "Dataset/TestDataset/CVC-300/images/151.png"
mask_path = "Dataset/TestDataset/CVC-300/masks/151.png"
save_path = None
target_layer = "model.decode_head.se_module"
num_chans = 128
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
        type='LAPFormerHead',
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

    # register hook
    layer = eval(target_layer)
    hook = IOHook(layer)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (352, 352))[:, :, ::-1]
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (352, 352))[:, :, 0]
    sample = transform(image=image, mask=mask)
    img, gt_mask = sample["image"], sample["mask"]
    gt_mask = np.asarray(gt_mask, np.float32)
    img = img[None].to(device)

    with torch.no_grad():
        model(img)
        feat_maps = hook.output
        if AVG:
            feat_maps = torch.mean(feat_maps, dim=1, keepdim=True)
        feat_maps = feat_maps.cpu().numpy()
    _, c, _, _ = feat_maps.shape
    print(f"Feature maps shape: {feat_maps.shape}")
    nrows, ncols = int(np.sqrt(num_chans)) or int(np.sqrt(c)), int(np.sqrt(num_chans)) or int(np.sqrt(c))

    if not AVG:
        num_c = 0
        fig_size = nrows * 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size, fig_size))
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].imshow(feat_maps[0, num_c, :, :])
                axes[i, j].axis('off')
                num_c += 1
    else:
        plt.imshow(feat_maps[0, 0])
        plt.axis('off')
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()