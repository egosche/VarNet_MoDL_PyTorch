
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn

class SSIMLoss(nn.Module):
    def __init__(self, win_size: int = 7,
                 K1: float = 0.01,
                 K2: float = 0.03,
                 full: bool = False):

        self.win_size = win_size
        self.K1 = K1
        self.K2 = K2
        self.full = full

    def forward(self, gt: torch.Tensor,
                pred: torch.Tensor,
                data_range: torch.Tensor):

        mssim = structural_similarity(gt, pred,
                                    data_range=data_range,
                                    win_size=self.win_size,
                                    K1=self.K1,
                                    K2=self.K2,
                                    full=self.full)

        return 1 - mssim
