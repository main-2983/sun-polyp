import torch.nn as nn

from mmcv.cnn import NORM_LAYERS


@NORM_LAYERS.register_module()
class ModifiedLayerNorm(nn.GroupNorm):
    def __init__(self,
                 num_channels,
                 **kwargs):
        super(ModifiedLayerNorm, self).__init__(num_groups=1,
                                                num_channels=num_channels,
                                                **kwargs)