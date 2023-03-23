from tqdm import tqdm
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from mcode.config_new_pipeline import *
import torch
from torch.utils.data import DataLoader
import wandb
from mmseg.models.builder import build_segmentor
from mcode import ActiveDataset, get_scores, LOGGER, set_logging
from mcode import select_device, ActiveDataset, UnNormalize, weighted_score
from tabulate import tabulate
import logging
import os
# config
save_img = True
save_path = "logs/MODEL_lapformer/infer"
show_img = True
ckpt_path = "runs/test/checkpoints_model_13/model_50.pth"
test_folder = "./Dataset/TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
        pretrained=None),
    decode_head=dict(
        type='LAPFormerHead_new_13',
        # ops='cat',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
)

def full_val(model, epoch):
    print("#" * 20)
    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice']
    ious, dices = AverageMeter(), AverageMeter()
    all_dices = []
    metric_weights = [0.1253, 0.0777, 0.4762, 0.0752, 0.2456]

    for dataset_name in dataset_names:
        data_path = f'{test_folder}/{dataset_name}'
        X_test = glob.glob('{}/images/*'.format(data_path))
        X_test.sort()
        y_test = glob.glob('{}/masks/*'.format(data_path))
        y_test.sort()

        test_dataset = ActiveDataset(X_test, y_test, transform3=val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        # print('Dataset_name:', dataset_name)
        gts = []
        prs = []
        for i, pack in enumerate(tqdm(test_loader), start=1):
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
        all_dices.append(mean_dice)
        if use_wandb:
            wandb.log({f'{dataset_name}_dice': mean_dice,
                       'epoch': epoch})
            wandb.log({f'{dataset_name}_iou': mean_iou,
                       'epoch': epoch})
        table.append([dataset_name, mean_iou, mean_dice])
    wdice = weighted_score(
        scores=all_dices,
        weights=metric_weights
    )
    table.append(['Total', ious.avg, dices.avg])
    table.append(['wDice', 0, wdice])
    if use_wandb:
        wandb.log({f'Avg_iou': ious.avg,
                   'epoch': epoch})
        wandb.log({f'Avg_dice': dices.avg,
                   'epoch': epoch})
        wandb.log({f'wDice': wdice,
                  'epoch': epoch})

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    with open(f"{save_path}/exp.log", 'a') as f:
        f.write(tabulate(table, headers=headers) + "\n")
    print("#" * 20)
    return ious.avg, dices.avg

if __name__ == '__main__':
    # init
    device = select_device('')

    # model
    model = build_segmentor(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
    model.to(device)
    model.eval()
    full_val(model, 50)
    # dataset
    # test_dataset = ActiveDataset(test_images, test_masks, transform3=transform)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False)
    # with torch.no_grad():
    #     for i, sample in tqdm(enumerate(test_loader)):
    #         image, gt, path = sample["image"], sample["mask"], sample["image_path"]
    #         _, dir = path[0].split(test_folder)
    #         _, dir, _, name = dir.split('/')
    #         gt = gt[0][0]
    #         gt = np.asarray(gt, np.float32)
    #         image = image.to(device)

    #         res = model(image)[0]
    #         res = res.sigmoid().data.cpu().numpy().squeeze()
    #         res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    #         pred = res.round()

    #         # visualize
    #         img = unorm(image.clone().squeeze(0))
    #         img = img.cpu().numpy().transpose(1, 2, 0)
    #         stacked = cv2.addWeighted(img, 0.5, np.repeat(np.expand_dims(pred, axis=-1), repeats=3, axis=-1), 0.5, 0)

    #         fig = plt.figure(figsize=(30, 10))
    #         plt.subplot(1, 3, 1)
    #         plt.imshow(stacked)
    #         plt.subplot(1, 3, 2)
    #         plt.imshow(img)
    #         plt.subplot(1, 3, 3)
    #         plt.imshow(gt)
    #         if save_img:
    #             if not os.path.exists(save_path):
    #                 os.makedirs(save_path, exist_ok=True)
    #             plt.savefig(f"{save_path}/P_{dir}_{name}.jpg")
    #         if show_img:
    #             plt.show()