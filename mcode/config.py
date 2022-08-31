import glob

import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from .utils import select_device
from .metrics import AverageMeter


# config
# ===============================================================================
use_wandb = True
wandb_key = None
wandb_project = "Seg-Uper"
wandb_entity = "ssl-online"
wandb_name = "B3-v5_defaultAug (1)"
wandb_group = "B3-v5_defaultAug"
wandb_dir = "./wandb"

seed = 2022
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_workers = 4

train_images = glob.glob('../Dataset/polyp/TrainDataset/image/*')
train_masks = glob.glob('../Dataset/polyp/TrainDataset/mask/*')

test_folder = "../Dataset/polyp/TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')

save_path = "runs/test"

image_size = 352

bs = 16
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
loss_weights = [0.8, 0.2]

train_transform = [
    {'Resize':{
        'size': [352, 352]
    }},
    {'RandomScaleCrop':{
        'range': [0.75, 1.25]
    }},
    {'RandomFlip':{
        'lr': True,
        'ud': True
    }},
    {'RandomRotate':{
        'range': [0, 359]
    }},
    {'RandomImageEnhance':{
        'methods': ['contrast', 'sharpness', 'brightness']
    }},
    {'RandomDilationErosion':{
        'kernel_range': [2, 5]
    }},
    {'ToNumpy': None},
    {'Normalize':{
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }},
    {'ToTensor': None}
]

val_transform = [
    {'Resize':{
        'size': [352, 352]
    }},
    {'ToNumpy': None},
    {'Normalize':{
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }},
    {'ToTensor': None}
]

pretrained = "/mnt/sdd/nguyen.van.quan/BKAI-kaggle/pretrained/mit_b1_mmseg.pth"
model_cfg = dict(
    type='SunSegmentor',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 18, 3],
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

# ===============================================================================