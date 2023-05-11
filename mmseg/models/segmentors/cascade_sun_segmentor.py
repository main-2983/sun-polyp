import torch
import torch.nn as nn

from mmseg.ops import resize

from ..builder import SEGMENTORS
from .sun_segmentor import SunSegmentor


@SEGMENTORS.register_module()
class CascadeSunSegmentor(SunSegmentor):
    def __init__(self,
                 **kwargs):
        super(CascadeSunSegmentor, self).__init__(**kwargs)
        self.check_cascade()

    def check_cascade(self):
        test_input = torch.rand((1, 3, 352, 352), device=next(self.parameters()).device)
        with torch.no_grad():
            out = self.forward(test_input)
        if not isinstance(out, list):
            raise AssertionError("Output is not a list, please use `SunSegmentor` instead of "
                                 "`CascadeSunSegmentor`")

    def forward(self, img):
        x = self.extract_feat(img)
        out = self.decode_head(x)
        out[0] = resize(
            input=out[0],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return out if self.training else out[0] # Output must have order of [/4, /8, /16, /32]
