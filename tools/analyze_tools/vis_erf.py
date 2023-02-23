import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mmseg.models.builder import build_segmentor

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mcode.metrics import AverageMeter
from mcode.utils import select_device
from mcode.dataset import ActiveDataset

save_path = "../../ERF_vis.npy"
transform = A.Compose([
    A.Resize(1024, 1024, interpolation=Image.BICUBIC),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
train_images = glob.glob('../Dataset/polyp/TestDataset/*/images/*')
train_masks = glob.glob('../Dataset/polyp/TestDataset/*/masks/*')
num_images = 50
ckpt_path = "../../logs/MLPOSA_v5/model_50.pth"
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

def get_input_grad(model, samples):
    outputs = model(samples)
    out_size = outputs.size()
    central_point = F.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = F.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


if __name__ == '__main__':
    device = select_device("" if torch.cuda.is_available() else 'cpu')

    # model
    model = build_segmentor(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    # dataset
    dataset = ActiveDataset(
        train_images,
        train_masks,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0 if device.type == 'cpu' else 2)

    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    meter = AverageMeter()
    optimizer.zero_grad()

    for batch_id, sample in enumerate(tqdm(dataloader), start=1):
        if meter.count == num_images:
            np.save(save_path, meter.avg)
            exit()
        n = sample["image"].shape[0]
        x = sample["image"].to(device)
        x.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, x)

        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print('accumulate')
            meter.update(contribution_scores)