import mmcv
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmseg.apis.inference import LoadImage, load_checkpoint, build_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.utils.util_distribution import get_device
import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import OrderedDict

from gen_mask import mask_to_rgb
from tqdm import tqdm

root_folder = "mitB1_uper_LRC"
config_file = f"logs/{root_folder}/main.py"
ckpt_file = f"logs/{root_folder}/model_50.pth"
split = "Kvasir"
folder = f"../Dataset/polyp/TestDataset/{split}/images"
mask_folder = f"../Dataset/polyp/TestDataset/{split}/masks"

PALETTE_MAPPING = {
    0: np.asarray([0, 0, 0]),
    1: np.asarray([255, 255, 255]),
}


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    #model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, OrderedDict):
        model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
        model.load_state_dict(state_dict)
    else:
        model = state_dict
    model.test_cfg = config.model['test_cfg']
    model.to(device)
    model.eval()
    return model


if __name__ == '__main__':
    device = get_device()
    cfg = Config.fromfile(config_file)
    model = init_segmentor(config_file, ckpt_file, device)
    for file in tqdm(os.listdir(folder)):
        name = file.split('.')[0]
        image_file = os.path.join(folder, file)
        mask_file = os.path.join(mask_folder, f"{name}.png")

        image = mmcv.imread(image_file)
        mask = mmcv.imread(mask_file)[:, :, 0]
        #rgb_mask = mask_to_rgb(mask, PALETTE_MAPPING)

        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=image)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        # forward the model
        with torch.no_grad():
            seg_logits = model.whole_inference(data['img'][0],
                                               data['img_metas'][0],
                                               rescale=True) # binary mask
            result = seg_logits.sigmoid().round().to(torch.int64).squeeze()
            #result = model(return_loss=False, rescale=True, **data)[0]  # binary mask
        palette = [0, 1]
        for i in range(len(palette)):
            palette[i] = PALETTE_MAPPING[i]

        seg = result
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = image * (1 - 0.3) + color_seg * 0.3
        img = img.astype(np.uint8)

        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(name)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_title("Pred")
        ax2.set_title("GT")
        im1 = ax1.imshow(img[:, :, ::-1].copy())
        im2 = ax2.imshow(mask)
        #plt.imshow(result[0].copy(), cmap=plt.cm.jet)
        #plt.show()
        plt.savefig(f'logs/{root_folder}/gen_masks/{split}/{name}.jpg')
        plt.close(fig)