from tqdm import tqdm
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

from mmseg.models.builder import build_segmentor

from mcode import select_device, ActiveDataset, UnNormalize

# config
save_img = True
save_path = "logs/MLPOSA_v5/infer"
show_img = True
ckpt_path = "logs/MLPOSA_v5/model_50.pth"
test_folder = "../Dataset/polyp/TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')
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

    # dataset
    test_dataset = ActiveDataset(test_images, test_masks, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)
    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader)):
            image, gt, path = sample["image"], sample["mask"], sample["image_path"]
            _, dir = path[0].split(test_folder)
            _, dir, _, name = dir.split('/')
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            image = image.to(device)

            res = model(image)[0]
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred = res.round()

            # visualize
            img = unorm(image.clone().squeeze(0))
            img = img.cpu().numpy().transpose(1, 2, 0)
            stacked = cv2.addWeighted(img, 0.5, np.repeat(np.expand_dims(pred, axis=-1), repeats=3, axis=-1), 0.5, 0)

            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(stacked)
            plt.subplot(1, 2, 2)
            plt.imshow(gt)
            if save_img:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/P_{dir}_{name}.jpg")
            if show_img:
                plt.show()