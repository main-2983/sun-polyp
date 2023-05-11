import glob

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

from .utils import select_device
from .metrics import AverageMeter
from .label_assignment import *


# config
# ===============================================================================

# wandb config
# ------------------------------------------------
use_wandb = False
wandb_key = None
wandb_project = "Seg-Uper"
wandb_entity = "ssl-online"
wandb_name = "TestGroup (2)"
wandb_group = None
wandb_dir = "./wandb"

# device config
# ------------------------------------------------
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_workers = 0

# data config
# ------------------------------------------------
train_images = glob.glob('Dataset/TrainDataset/image/*')
train_masks = glob.glob('Dataset/TrainDataset/mask/*')

test_folder = "Dataset/TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')

image_size = 352

bs = 2

save_path = "runs/test"

# running statistic
# ------------------------------------------------
train_loss_meter = AverageMeter()
iou_meter = AverageMeter()
dice_meter = AverageMeter()

# epoch config
# ------------------------------------------------
n_eps = 50
save_ckpt_ep = 40
val_ep = 40
best = -1.

# optimizer
# ------------------------------------------------
use_SAM = False
optimizer = torch.optim.AdamW
init_lr = 1e-4
grad_accumulate_rate = 1
optimizer_kwargs = dict(
    lr=init_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# loss config
# ------------------------------------------------
loss_cfg = dict(
    type='StructureLoss',
    weight=1.0
)

# augmentation
# ------------------------------------------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomGamma (gamma_limit=(50, 150), eps=None, always_apply=False, p=0.5),
    # A.RandomBrightness(p=0.3),
    # A.RGBShift(p=0.3, r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    # A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
    # A.Cutout(p=0.3, max_h_size=25, max_w_size=25, fill_value=255),
    # A.ShiftScaleRotate(p=0.3, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.11),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# deep supervision
# ------------------------------------------------
strategy = cascade_target # default to None
strategy_kwargs = dict(
    num_outs=3,
    mode='bilinear'
)
label_vis_kwargs = dict(
    type="iter",

)

# model config
# ------------------------------------------------
pretrained = 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_4s-06e79181.pth'
model_cfg = dict(
    type='CascadeSunSegmentor',
    backbone=dict(
        type='TIMMBackbone',
        model_name='res2net50_26w_4s',
        out_indices=[1, 2, 3, 4],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained
        )
    ),
    decode_head=dict(
        type='PraNetHead',
        num_classes=1,
        in_index=[1, 2, 3],
        in_channels=[512, 1024, 2048],
        channels=256,
        norm_cfg=dict(type='BN',
                      requires_grad=True)
    )
)

# ===============================================================================