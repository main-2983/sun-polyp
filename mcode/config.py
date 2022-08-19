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
wandb_key = "d0ee13baa7af4379eff80e68b11cf976bbb8d673"
wandb_project = "Seg-Uper"
wandb_entity = "ssl-online"
wandb_name = "TestGroup (2)"
wandb_group = None
wandb_dir = "./wandb"

seed = 2022
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')
num_workers = 4

train_images = glob.glob('../Dataset/polyp/TrainDataset/images/*')
train_masks = glob.glob('../Dataset/polyp/TrainDataset/masks/*')

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

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomGamma (gamma_limit=(50, 150), eps=None, always_apply=False, p=0.5),
    A.RandomBrightness(p=0.3),
    A.RGBShift(p=0.3, r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
    A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
    A.Cutout(p=0.3, max_h_size=25, max_w_size=25, fill_value=255),
    A.ShiftScaleRotate(p=0.3, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.11),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ===============================================================================