import glob

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from .utils import select_device
from .metrics import AverageMeter
from .label_assignment import *


name_model = "LAPFormerHead"
name_wandb = "LAPFormerHead"
# config
# ===============================================================================
use_wandb = False
wandb_key = "1424c55fa73c0d5684ab0210260f866920bb498d"
wandb_project = "Polyp-Research"
wandb_entity = "ssl-online"
wandb_name = '0'
wandb_group = name_wandb
# wandb_dir = "~/wandb"

seed = 2022
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_workers = 4

train_images = glob.glob('Dataset/TrainDataset/image/*')
train_masks = glob.glob('Dataset/TrainDataset/mask/*')

test_folder = "Dataset/TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')

save_path = "runs/test"

image_size = 352

bs = 1
bs_val = 2
grad_accumulate_rate = 1

train_loss_meter = AverageMeter()
iou_meter = AverageMeter()
dice_meter = AverageMeter()

n_eps = 50
save_ckpt_ep = 40
val_ep = 40
best = -1.

init_lr = 1e-4

focal_loss = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
bce_loss = smp.losses.SoftBCEWithLogitsLoss()
loss_fns = [bce_loss, dice_loss]
loss_weights = [0.5, 0.5]

transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

strategy = None # default to None
strategy_kwargs = {

}
label_vis_kwargs = {
    'type': None
}

pretrained = "pretrained/mit_b1_mmseg.pth"
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
        pretrained=pretrained),
    decode_head=dict(
        type='LAPFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

# ===============================================================================