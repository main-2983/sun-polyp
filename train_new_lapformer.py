import wandb
from tqdm import tqdm
from tabulate import tabulate
import logging
import os
from torchinfo import summary
import torch
from utils_polyp_pvt.utils import clip_gradient
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
from mmseg.models.builder import build_segmentor

from mcode import ActiveDataset, get_scores, LOGGER, set_logging
from mcode.config import *


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


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

        test_dataset = ActiveDataset(X_test, y_test, transform=transform)
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

            res = model(image)[0]
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

    # Log config
    with open("mcode/config.py", 'r') as f:
        config_data = f.read().strip()
        with open(f"{save_path}/exp.log", 'w') as log_f:
            log_f.write(f"{config_data} \n")

    # set_seed_everything(seed)

    if use_wandb:
        assert wandb_group is not None, "Please specify wandb group"
        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            # dir=wandb_dir,
            group=wandb_group
        )

    # model
    model = build_segmentor(model_cfg)
    model.init_weights()
    model = model.to(device)
    summary(model, input_size=(1,3,352,352))
    # from pthflops import count_ops
    # inp = torch.rand(1,3,352,352).to(device)
    # count_ops(model, inp)
    # dataset
    train_dataset = ActiveDataset(
        train_images,
        train_masks,
        transform=transform
    )
    # val_dataset = ActiveDataset(
    #     test_images,
    #     test_masks,
    #     trainsize=image_size,
    #     transform=val_transform
    # )

    set_logging("Polyp")
    LOGGER = logging.getLogger("Polyp")
    LOGGER.info(f"Train size: {len(train_dataset)}")
    # LOGGER.info(f"Valid size: {len(val_dataset)}")

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=True)
    total_step = len(train_loader)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=len(train_loader) * n_eps,
                                                              eta_min=init_lr / 1000)

    with open(f"{save_path}/exp.log", 'a') as f:
        f.write("Start Training...\n")

    for ep in range(1, n_eps + 1):
        size_rates = [0.75, 1, 1.25]
        dice_meter.reset()
        iou_meter.reset()
        train_loss_meter.reset()
        model.train()

        for batch_id, sample in enumerate(tqdm(train_loader), start=1):
            if ep <= 1:
                optimizer.param_groups[0]["lr"] = (ep * batch_id) / (1.0 * total_step) * init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates:
                n = sample['image'].shape[0]
                image = sample['image'].to(device)
                gts = sample['mask'].to(device)

                trainsize = int(round(image_size*rate/32)*32)
                x = F.upsample(image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                y = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                y_hats = model(x)
                losses = []
                for y_hat in y_hats:
                    loss = structure_loss(y_hat, y)
                    losses.append(loss)
                losses = sum(_loss for _loss in losses)
                losses.backward()
                clip_gradient(optimizer, 0.5)
                if batch_id % grad_accumulate_rate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if rate == 1:
                    y_hat_mask = y_hats[0].sigmoid()
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

    if use_wandb:
        wandb.save(f"{save_path}/exp.log")