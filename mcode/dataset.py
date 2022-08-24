import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .augmentation import *

class ActiveDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths=[], gt_paths=[], transform_list=None):
        assert len(image_paths) > 0, "Can't find any images in dataset"
        assert len(gt_paths) > 0, "Can't find any mask in dataset"
        self.images = image_paths
        self.masks = gt_paths
        self.size = len(self.images)
        self.filter_files()
        self.transform = self.get_transform(transform_list)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])

        shape = mask.size[::-1]
        sample = {'image': image, 'mask': mask, 'shape': shape}
        if self.transform is not None:
            sample = self.transform(sample)
            image = sample["image"]
            mask = sample["mask"]

            image = torch.as_tensor(np.array(image))
            mask = torch.as_tensor(np.array(mask))

        sample = dict(image=image, mask=mask, image_path=self.images[index], mask_path=self.masks[index])

        return sample

    def get_transform(self, transform_list):
        tfs = []
        for transform_dict in transform_list:
            for key, value in zip(transform_dict.keys(), transform_dict.values()):
                if value is not None:
                    tf = eval(key)(**value)
                else:
                    tf = eval(key)()
                tfs.append(tf)
        return transforms.Compose(tfs)

    def filter_files(self):
        assert len(self.images) == len(self.masks)
        images = []
        masks = []
        for img_path, mask_path in zip(self.images, self.masks):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size == mask.size:
                images.append(img_path)
                masks.append(mask_path)
        self.images = images
        self.masks = masks

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            return img

    def __len__(self):
        return self.size


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
