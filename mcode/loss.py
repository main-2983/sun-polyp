from typing import List

import torch.nn as nn
from mmseg.models import build_loss


def make_loss(loss_cfg) -> List[nn.Module]:
    assert isinstance(loss_cfg, (dict, list))
    if isinstance(loss_cfg, dict):
        loss = build_loss(loss_cfg)
        loss = [loss]
    else:
        loss = []
        for cfg in loss_cfg:
            loss.append(build_loss(cfg))
    return loss
