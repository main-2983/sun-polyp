from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import glob
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ActiveDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths=[], gt_paths=[], is_test = False, trainsize=352, transform1=None, transform2=None):
        self.trainsize = trainsize
        assert len(image_paths) > 0, "Can't find any images in dataset"
        assert len(gt_paths) > 0, "Can't find any mask in dataset"
        self.images = image_paths
        self.masks = gt_paths
        self.size = len(self.images)
        self.filter_files()
        self.is_test = is_test
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])

        if self.transform1 is not None:
            transformed = self.transform1(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask / 255
        if self.is_test:
            return dict(image=image, mask=mask.unsqueeze(0), image_path=self.images[index], mask_path=self.masks[index])  

        if self.transform2 is not None:
            pseudo_mask = np.zeros((self.trainsize, self.trainsize))
            transformed = self.transform2(image=image, mask=mask)
            mask = transformed["mask"]
            image1 = transformed["image"]
            transformed = self.transform2(image=image, mask=pseudo_mask)
            image2 = transformed["image"]
            
        sample = dict(image1=image1, image2=image2, mask=mask.unsqueeze(0), image_path=self.images[index], mask_path=self.masks[index])

        return sample

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
            img = Image.open(f).resize((self.trainsize, self.trainsize), Image.BILINEAR)
            return np.array(img.convert('RGB'))

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((self.trainsize, self.trainsize), Image.NEAREST)
            img = np.array(img.convert('L'))
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

# if __name__== "__main__":
#     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#     train_transform_1 = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.ShiftScaleRotate(p=0.3, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.11),
#     ])

#     train_transform_2 = A.Compose([
#         A.RandomGamma (gamma_limit=(50, 150), eps=None, always_apply=False, p=0.5),
#         A.RandomBrightness(p=0.3),
#         A.RGBShift(p=0.3, r_shift_limit=5, g_shift_limit=5, b_shift_limit=5),
#         A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
#         # A.Cutout(p=0.3, max_h_size=25, max_w_size=25, fill_value=255),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])
#     train_images = glob.glob('../Dataset/TrainDataset/image/*')
#     train_masks = glob.glob('../Dataset/TrainDataset/mask/*')
#     demo_data = ActiveDataset(train_images, train_masks, 352, train_transform_1, train_transform_2)
#     import random
#     import matplotlib.pyplot as plt
#     idx = random.randint(0, 1000)
#     sample = demo_data.__getitem__(idx)
#     x1, x2, y = sample["image1"], sample["image2"], sample["mask"]
#     cv2.imwrite("img1.png", unorm(x1).permute(1,2,0).numpy()[:,:,::-1]*255)
    
#     cv2.imwrite("img2.png", unorm(x2).permute(1,2,0).numpy()[:,:,::-1]*255)

#     cv2.imwrite("img3.png", y.squeeze(0).numpy()*255)