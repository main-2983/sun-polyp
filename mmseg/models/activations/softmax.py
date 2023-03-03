import torch.nn as nn

from mmcv.cnn import ACTIVATION_LAYERS


ACTIVATION_LAYERS.register_module(module=nn.Softmax, name='Softmax')