import glob
import wandb
from tqdm import tqdm
from tabulate import tabulate
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from mcode import model, AverageMeter, get_scores, ActiveDataset, set_seed_everything,\
    LOGGER, select_device, set_logging

# config
# ===============================================================================
use_wandb = False
wandb_key = "d0ee13baa7af4379eff80e68b11cf976bbb8d673"
wandb_project = "Seg-Uper"
wandb_entity = "ssl-online"
wandb_name = "B1-eSE_eLA_RA"
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

bs = 2
bs_val = 2
grad_accumulate_rate = 2

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

def full_val(model):
    print("#" * 20)
    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice']
    ious, dices = AverageMeter(), AverageMeter()

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        X_test = glob.glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob.glob('{}/masks/*'.format(data_path))
        y_test.sort()

        test_dataset = ActiveDataset(X_test, y_test, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        # print('Dataset_name:', dataset_name)
        gts = []
        prs = []
        for i, pack in enumerate(test_loader, start=1):
            image, gt = pack["image"], pack["mask"]
            gt = gt[0][0]
            gt = np.asarray(gt, np.float32)
            image = image.to(device)

            res = model.forward_dummy(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()
            gts.append(gt)
            prs.append(pr)
        mean_iou, mean_dice, _, _ = get_scores(gts, prs)
        ious.update(mean_iou)
        dices.update(mean_dice)
        if use_wandb:
            wandb.log({f'{dataset_name}_dice': mean_dice})
            wandb.log({f'{dataset_name}_iou': mean_iou})
        table.append([dataset_name, mean_iou, mean_dice])
    table.append(['Total', ious.avg, dices.avg])

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    with open(f"{save_path}/exp.log", 'a') as f:
        f.write(tabulate(table, headers=headers) + "\n")
    print("#" * 20)
    return ious.avg, dices.avg


if __name__ == '__main__':
    # Create log folder
    if not os.path.exists(f"{save_path}/checkpoints"):
        os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
    LOGGER.info(f"Experiment will be saved to {save_path}")

    # Log model config
    with open("mcode/model.py", 'r') as f:
        model_data = f.read().strip()
        with open(f"{save_path}/exp.log", 'w') as log_f:
            log_f.write(f"+ MODEL CONFIG \n {model_data} \n")

    set_seed_everything(seed)
    if use_wandb:
        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            dir=wandb_dir
        )

    # model
    model = model()
    model = model.to(device)

    # dataset
    train_dataset = ActiveDataset(
        train_images,
        train_masks,
        trainsize=image_size,
        transform=train_transform
    )
    val_dataset = ActiveDataset(
        test_images,
        test_masks,
        trainsize=image_size,
        transform=val_transform
    )

    set_logging("Polyp")
    LOGGER = logging.getLogger("Polyp")
    LOGGER.info(f"Train size: {len(train_dataset)}")
    LOGGER.info(f"Valid size: {len(val_dataset)}")

    # Log data settings
    with open(f"{save_path}/exp.log", 'a') as f:
        f.write(f"+ TRAIN TRANSFORM \n {str(train_transform)} \n")
        f.write(f"+ VAL TRANSFORM \n {str(val_transform)} \n")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers)
    total_step = len(train_loader)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=len(train_loader) * n_eps,
                                                              eta_min=init_lr / 1000)

    with open(f"{save_path}/exp.log", 'a') as f:
        f.write("Start Training...\n")

    for ep in range(1, n_eps + 1):
        dice_meter.reset()
        iou_meter.reset()
        train_loss_meter.reset()
        model.train()

        for batch_id, sample in enumerate(tqdm(train_loader), start=1):
            if ep <= 1:
                optimizer.param_groups[0]["lr"] = (ep * batch_id) / (1.0 * total_step) * init_lr
            else:
                lr_scheduler.step()

            n = sample["image"].shape[0]
            x = sample["image"].to(device)
            y = sample["mask"].to(device).to(torch.int64)
            y_hat = model.forward_dummy(x)
            loss = loss_weights[0] * loss_fns[0](y_hat.squeeze(1), y.squeeze(1).float()) + \
                   loss_weights[1] * loss_fns[1](y_hat, y)
            loss.backward()

            if batch_id % grad_accumulate_rate == 0:
                optimizer.step()
                optimizer.zero_grad()
            y_hat_mask = y_hat.sigmoid()
            pred_mask = (y_hat_mask > 0.5).float()

            train_loss_meter.update(loss.item(), n)
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), y.long(), mode="binary")
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            iou_meter.update(per_image_iou, n)
            dice_meter.update(dataset_iou, n)

        LOGGER.info("EP {} TRAIN: LOSS = {}, avg_dice = {}, avg_iou = {}".format(ep, train_loss_meter.avg, dice_meter.avg,
                                                                           iou_meter.avg))

        # Log metrics
        with open(f"{save_path}/exp.log", 'a') as f:
            f.write("EP {} TRAIN: LOSS = {}, avg_dice = {}, avg_iou = {} \n".format(ep, train_loss_meter.avg, dice_meter.avg,
                                                                           iou_meter.avg))

        if use_wandb:
            wandb.log({'train_dice': dice_meter.avg})
        if ep >= save_ckpt_ep:
            torch.save(model.state_dict(), f"{save_path}/checkpoints/model_{ep}.pth")

        if ep >= val_ep:
            # val model
            with torch.no_grad():
                iou, dice = full_val(model)
                if (dice > best):
                    torch.save(model.state_dict(), f"{save_path}/checkpoints/best.pth")
                    best = dice

            print("================================\n")
