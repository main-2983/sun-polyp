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
use_wandb = False
wandb_key = None
wandb_project = "Seg-Uper"
wandb_entity = "Polyp-Research"
wandb_name = "RFP (1)"
wandb_group = "RFP"
wandb_dir = "./wandb"

seed = 2022
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_workers = 4

train_images = glob.glob('/mnt/sdd/nguyen.van.quan/Researchs/Polyp/TrainDataset/image/*')
train_masks = glob.glob('/mnt/sdd/nguyen.van.quan/Researchs/Polyp/TrainDataset/mask/*')

test_folder = "/mnt/sdd/nguyen.van.quan/Researchs/Polyp/TestDataset"
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

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

pretrained = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_4s-06e79181.pth'
model_cfg = dict(
    type='SunSegmentor',
    backbone=dict(
        type='TIMMBackbone',
        model_name='res2net50_26w_4s',
        out_indices=[1, 2, 3, 4],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    decode_head=dict(
        type='FCNHead',
        num_classes=1,
        input_transform='resize_concat',
        in_index=[0, 1, 2, 3],
        in_channels=[256, 256, 256, 256],
        channels=256
    )
)

# ===============================================================================