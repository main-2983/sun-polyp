import wandb
from tqdm import tqdm
from tabulate import tabulate
import logging
import os
from torchinfo import summary
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from mmseg.models.builder import build_segmentor
from mcode.sam import SAM
from mcode import ActiveDataset, get_scores, LOGGER, set_seed_everything, set_logging
from mcode.config import *

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

        test_dataset = ActiveDataset(X_test, y_test, is_test = True, transform1=val_transform)
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

    # from pthflops import count_ops
    # inp = torch.rand(1,3,352,352).to(device)
    # count_ops(model, inp)
    # dataset
    train_dataset = ActiveDataset(
        train_images,
        train_masks,
        is_test = False,
        trainsize=image_size,
        transform1=train_transform_1,
        transform2=train_transform_2,
        transform3=train_transform3
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
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers, shuffle=True)
    total_step = len(train_loader)

        # model
    model = build_segmentor(model_cfg)
    model.init_weights()
    model = model.to(device)
    summary(model, input_size=(1,3,352,352))

    # optimizer
    if use_SAM:
        LOGGER.warning("You're using SAM for training, training will be slower than usual")
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optimizer,
            **optimizer_kwargs
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer,
                                                                  T_max=len(train_loader) * n_eps,
                                                                  eta_min=init_lr / 1000)
    else:
        optimizer = optimizer(
            model.parameters(),
            **optimizer_kwargs
        )
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
        # if not os.path.exists(f"{seed}_{name_wandb}/ep_{ep}"):
        #     os.makedirs(f"{seed}_{name_wandb}/ep_{ep}")
        for batch_id, sample in enumerate(tqdm(train_loader), start=1):
            # if ep in [1, 2, 3] and batch_id < 91:
            #     for i in range(bs):
            #         img_demo = unorm(sample["image"][i]).permute(1, 2, 0).numpy() * 255
            #         img_demo = cv2.cvtColor(img_demo, cv2.COLOR_BGR2RGB)
            #         cv2.imwrite(f"{seed}_{name_wandb}/ep_{ep}/ep_1_image_id_{batch_id}_num_{i}.png", img_demo)
            if ep <= 1:
                optimizer.param_groups[0]["lr"] = (ep * batch_id) / (1.0 * total_step) * init_lr
            else:
                lr_scheduler.step()

            n = sample["image1"].shape[0]
            x1, x2 = sample["image1"].to(device), sample["image2"].to(device)
            y = sample["mask"].to(device).to(torch.int64)
            y_hats1 = model(x1)
            y_hats2 = model(x2)
            y_hats = y_hats1+y_hats2
            losses = []
            for y_hat in y_hats:
                loss = loss_weights[0] * loss_fns[0](y_hat.squeeze(1), y.squeeze(1).float()) + \
                       loss_weights[1] * loss_fns[1](y_hat, y)
                losses.append(loss)
            losses = sum(_loss for _loss in losses)
            aux_loss = loss_fns[2](y_hats1[0].sigmoid(), y_hats2[0].sigmoid())
            losses = losses + alpha*aux_loss
            losses.backward()

            def closure():
                _y_hats1 = model(x1)
                _y_hats2 = model(x2)
                _y_hats = _y_hats1 + _y_hats2
                _losses = []
                for _y_hat in _y_hats:
                    _loss = loss_weights[0] * loss_fns[0](_y_hat.squeeze(1), y.squeeze(1).float()) + \
                        loss_weights[1] * loss_fns[1](_y_hat, y)
                    _losses.append(_loss)
                _losses = sum(_l for _l in _losses)

                _aux_loss = loss_fns[2](_y_hats1[0].sigmoid(), _y_hats2[0].sigmoid())
                _losses = _losses + alpha*_aux_loss
                _losses.backward()
                return _losses

            if batch_id % grad_accumulate_rate == 0:
                optimizer.step(closure=closure if use_SAM else None)
                optimizer.zero_grad()
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
            print(f"saved model_{ep}.pth")

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
