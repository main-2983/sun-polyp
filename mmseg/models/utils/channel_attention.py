import torch.nn as nn


__all__ = [
    'ECAModule'
]


class ECAModule(nn.Module):
    """
    ECA Module from ECA-Net
    References: https://arxiv.org/pdf/1910.03151.pdf
    Args:
        kernel_size: window for mixing channel
    kernel_size should be int(abs((log(C, 2) + 1 / 2)) + 1
    """
    def __init__(self,
                 kernel_size=3):
        super(ECAModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.pool(x) # b, c, 1, 1
        weight = weight.squeeze(-1).permute(0, 2, 1) # b, 1, c
        weight = self.conv(weight)
        weight = self.sigmoid(weight)
        weight = weight.permute(0, 2, 1).unsqueeze(-1) # b, c, 1, 1
        return weight * x
