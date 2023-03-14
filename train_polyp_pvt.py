import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils_polyp_pvt.dataloader import get_loader, test_dataset
from utils_polyp_pvt.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import tqdm
import logging
import torch.nn as nn
from tabulate import tabulate
from mcode.config import *
from mmseg.models.builder import build_segmentor
from torchinfo import summary
from mcode import get_scores

import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


# class FocalLossV1(nn.Module):

#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2,
#                  reduction='mean',):
#         super(FocalLossV1, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.crit = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, label):
#         # compute loss
#         logits = logits.float() # use fp32 if logits is fp16
#         with torch.no_grad():
#             alpha = torch.empty_like(logits).fill_(1 - self.alpha)
#             alpha[label == 1] = self.alpha

#         probs = torch.sigmoid(logits)
#         pt = torch.where(label == 1, probs, 1 - probs)
#         ce_loss = self.crit(logits, label.float())
#         loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wfocal = FocalLossV1()(pred, mask)
#     wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wfocal + wiou).mean()


def full_val(model):
    print("#" * 20, "start testing", "#" * 20)
    model.eval()
    dataset_names = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    table = []
    headers = ['Dataset', 'IoU', 'Dice']
    ious, dices = AverageMeter(), AverageMeter()

    for dataset_name in tqdm.tqdm(dataset_names):
        data_path = f'{test_folder}/{dataset_name}'
        X_test = os.path.join(data_path, 'images/')
        Y_test = os.path.join(data_path, 'masks/')
        TestDataset = test_dataset(X_test, Y_test, 352)
        # for i in TestDataset:
        #     print(i[0].shape)
        test_loader = torch.utils.data.DataLoader(
            TestDataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        # print('Dataset_name:', dataset_name)
        gts = []
        prs = []
        for i, pack in enumerate(test_loader, start=1):
            image, gt = pack
            gt = gt[0]
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)

            res = model(image)[0]
            res = F.upsample(res.unsqueeze(0) , size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pr = res.round()
            gts.append(gt)
            prs.append(pr)
        mean_iou, mean_dice, _, _ = get_scores(gts, prs)
        ious.update(mean_iou)
        dices.update(mean_dice)
        table.append([dataset_name, mean_iou, mean_dice])
    table.append(['Total', ious.avg, dices.avg])

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return ious.avg, dices.avg



def train(train_loader, model, optimizer, epoch):
    model.train()
    size_rates = [0.75, 1, 1.25] 
    loss_P1_record = AvgMeter()
    for i, pack in enumerate(tqdm.tqdm(train_loader), start=1):
        # if epoch <= 1:
        #         optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * init_lr
        # else:
        #     lr_scheduler.step()

        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1 = model(images)[0]

            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss = loss_P1 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
        # ---- train visualization ----
        if i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' Loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
    
if __name__ == '__main__':
    name = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
    ##################model_name#############################
    model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./Dataset/TrainDataset',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = build_segmentor(model_cfg)
    model.init_weights()
    model = model.to(device)
    summary(model, input_size=(1,3,352,352))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    
    print(optimizer)

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    if os.path.exists(image_root):

        train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                                augmentation=opt.augmentation)
        total_step = len(train_loader)

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        #                             T_max=len(train_loader)*opt.epoch,
        #                             eta_min=init_lr/1000)

        print("#" * 20, "Start Training", "#" * 20)

        for epoch in range(1, opt.epoch):
            print('epoch: ', epoch)
            adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
            train(train_loader, model, optimizer, epoch)
            if epoch >= 40:
                full_val(model)
        # plot the eval.png in the training stage
        # plot_train(dict_plot, name)
    else:
        print("Can't find any images in dataset")
