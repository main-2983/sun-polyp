import glob
import wandb
from tqdm import tqdm
from tabulate import tabulate

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from code.model import model
from code.metrics import AverageMeter, get_scores
from code.dataset import ActiveDataset
from code.utils import seed_everything, LOGGER, select_device

# config
# ===============================================================================
wandb_key = "d0ee13baa7af4379eff80e68b11cf976bbb8d673"
wandb_project = "Seg-Uper"
wandb_entity = "ssl-online"
wandb_name = "B1-eSE_eLA_RA"
wandb_dir = "./wandb"

seed = 2022
device = select_device("cuda:0" if torch.cuda.is_available() else 'cpu')

train_images = glob.glob('TrainDataset/image/*')
train_masks = glob.glob('TrainDataset/mask/*')

test_images = glob.glob('TestDataset/*/images/*')
test_masks = glob.glob('TestDataset/*/masks/*')

save_path = "/content/polyp/checkpoints"

image_size = 352

bs = 8
bs_val = 4

train_loss_meter = AverageMeter()
iou_meter = AverageMeter()
dice_meter = AverageMeter()

n_eps = 50
best = -1.

init_lr = 1e-4

focalloss = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
diceloss = smp.losses.DiceLoss(smp.losses.BINARY_MODE)
loss_bce = smp.losses.SoftBCEWithLogitsLoss()

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

def inference(model):
    print("#" * 20)
    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice']
    ious, dices = AverageMeter(), AverageMeter()

    for dataset_name in dataset_names:
        data_path = f'TestDataset/{dataset_name}'
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
        wandb.log({f'{dataset_name}_dice': mean_dice})
        wandb.log({f'{dataset_name}_iou': mean_iou})
        table.append([dataset_name, mean_iou, mean_dice])
    table.append(['Total', ious.avg, dices.avg])

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    print("#" * 20)
    return ious.avg, dices.avg


if __name__ == '__main__':
    seed_everything(seed)
    wandb.login(key=wandb_key)
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
        dir=wandb_dir
    )

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
    LOGGER.info(f"Train size: {len(train_dataset)}")
    LOGGER.info(f"Valid size: {len(val_dataset)}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4)
    total_step = len(train_loader)

    # model
    model = model()
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=len(train_loader) * n_eps,
                                                              eta_min=init_lr / 1000)

    for ep in range(n_eps):

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
            loss = 0.8 * loss_bce(y_hat.squeeze(1), y.squeeze(1).float()) + 0.2 * diceloss(y_hat, y)
            loss.backward()

            if batch_id % 2 == 0:
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
        wandb.log({'train_dice': dice_meter.avg})
        if ep >= int(1/2 * n_eps):
            torch.save(model.state_dict(), f"{save_path}/model_{ep}.pth")

        if ep >= int(2/3 * n_eps):
            # val model
            with torch.no_grad():
                iou, dice = inference(model)
                if (dice > best):
                    torch.save(model.state_dict(), f"{save_path}/best.pth")
                    best = dice
                    LOGGER.info("saved best: ", dice)

            print("================================\n")
