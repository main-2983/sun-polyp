import cv2
import os
from tqdm import tqdm

import torch
from torch import nn
import mmcv
from mmcv.cnn import ConvModule

root_folder_a = "mitB1_MLP"
root_folder_b = "mitB1_MLPSlow"
root_folder_c = "mitB1_MLPSlow_VA"
root_folder_d = "mitB1_uper_LRC"
split = "Kvasir"
folder_a = f"logs/{root_folder_a}/gen_masks/{split}"
folder_b = f"logs/{root_folder_b}/gen_masks/{split}"
folder_c = f"logs/{root_folder_c}/gen_masks/{split}"
folder_d = f"logs/{root_folder_d}/gen_masks/{split}"


if __name__ == '__main__':
    for file in os.listdir(folder_a):
        print(file)
        file_path_a = os.path.join(folder_a, file)
        file_path_b = os.path.join(folder_b, file)
        file_path_c = os.path.join(folder_c, file)
        file_path_d = os.path.join(folder_d, file)

        image_a = cv2.imread(file_path_a)
        image_b = cv2.imread(file_path_b)
        image_c = cv2.imread(file_path_c)
        image_d = cv2.imread(file_path_d)

        cv2.namedWindow("MLP")
        cv2.namedWindow("MLPSlow")
        cv2.namedWindow("MLPSlow+VA")
        cv2.namedWindow("UperLRC")

        cv2.imshow("MLP", image_a)
        cv2.imshow("MLPSlow", image_b)
        cv2.imshow("MLPSlow+VA", image_c)
        cv2.imshow("UperLRC", image_d)

        cv2.waitKey(0)

