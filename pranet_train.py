import wandb
from tqdm import tqdm
from tabulate import tabulate
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from mmseg.models.builder import build_segmentor
from mmseg.models.losses import StructureLoss

from mcode import ActiveDataset, get_scores, LOGGER, set_seed_everything, set_logging
from mcode.utils import adjust_lr, clip_gradient
from mcode.config import *


def full_val(model, epoch):
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
            wandb.log({f'{dataset_name}_dice': mean_dice,
                       'epoch': epoch})
            wandb.log({f'{dataset_name}_iou': mean_iou,
                       'epoch': epoch})
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

    set_seed_everything(seed)
    if use_wandb:
        assert wandb_group is not None, "Please specify wandb group"
        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            dir=wandb_dir,
            group=wandb_group
        )

    # model
    model = build_segmentor(model_cfg)
    model.init_weights()
    model = model.to(device)

    # loss
    loss_fn = StructureLoss()

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

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers)
    total_step = len(train_loader)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # label visualize
    label_vis_hook = LabelVis(model, save_path)
    # --- before train hooks ---
    label_vis_hook.before_train(train_dataset)

    for ep in range(1, n_eps + 1):
        # this fucking line literally do nothing ????????????????
        adjust_lr(optimizer, 1e-4, ep, 0.1, 50)

        dice_meter.reset()
        iou_meter.reset()
        train_loss_meter.reset()
        model.train()

        # --- multi-scale training ---
        size_rates = [0.75, 1, 1.25]
        for batch_id, sample in enumerate(tqdm(train_loader), start=1):
            for rate in size_rates:
                # --- data prepare ---
                n = sample["image"].shape[0]
                x = sample["image"].to(device)
                y = sample["mask"].to(device)
                # --- rescale ---
                image_size = int(round(image_size * rate / 32) * 32)
                if rate != 1:
                    x = F.upsample(x, size=(image_size, image_size),
                                   mode='bilinear', align_corners=True)
                    y = F.upsample(y, size=(image_size, image_size),
                                   mode='bilinear', align_corners=True)
                # --- forward ---
                y_hats = model(x)
                # --- get targets ---
                #strategy_kwargs['cur_ep'] = ep # uncomment this if not strategy 2
                targets = label_assignment(y_hats, y, strategy, **strategy_kwargs)
                # --- loss function ---
                losses = []
                if len(loss_weights) == 1:
                    loss_weights = loss_weights * len(y_hats)
                for i, (y_hat, y) in enumerate(zip(y_hats, targets)):
                    loss = loss_fn(y_hat, y)
                    loss = loss_weights[i] * loss
                    losses.append(loss)
                losses = sum(_loss for _loss in losses)
                # --- backward ---
                losses.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()
                optimizer.zero_grad()
                # --- evaluate on train dataset ---
                y_hat_mask = y_hats[0].sigmoid()
                pred_mask = (y_hat_mask > 0.5).float()

                train_loss_meter.update(loss.item(), n)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), targets[0].long(), mode="binary")
                per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
                dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                iou_meter.update(per_image_iou, n)
                dice_meter.update(dataset_iou, n)

            # --- after train iter hooks ---
            label_vis_hook.after_train_iter(batch_id, ep, strategy_kwargs)
        # --- after train epoch hooks ---
        label_vis_hook.after_train_epoch(ep, strategy_kwargs)

        LOGGER.info(
            "EP {} TRAIN: LOSS = {}, avg_dice = {}, avg_iou = {}".format(ep, train_loss_meter.avg, dice_meter.avg,
                                                                         iou_meter.avg))

        # Log metrics
        with open(f"{save_path}/exp.log", 'a') as f:
            f.write("EP {} TRAIN: LOSS = {}, avg_dice = {}, avg_iou = {} \n".format(ep, train_loss_meter.avg,
                                                                                    dice_meter.avg,
                                                                                    iou_meter.avg))

        if use_wandb:
            wandb.log({'train_dice': dice_meter.avg,
                       'epoch': ep})
        if ep >= save_ckpt_ep:
            torch.save(model.state_dict(), f"{save_path}/checkpoints/model_{ep}.pth")

        if ep >= val_ep:
            # val model
            with torch.no_grad():
                iou, dice = full_val(model, ep)
                if (dice > best):
                    torch.save(model.state_dict(), f"{save_path}/checkpoints/best.pth")
                    best = dice

            print("================================\n")

    if use_wandb:
        wandb.save(f"{save_path}/exp.log")

