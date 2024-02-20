"""
Variational Network

Reference:
* Hammernik K, Klatzer T, Kobler E, Recht MP, Sodickson DK, Pock T, Knoll F. Learning a variational network for reconstruction of accelerated MRI data. Magn Reson Med 2018;79:3055-3071.

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from utils import r2c, c2r
from models import mri, unet2d

# %%
class data_consistency(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lam = nn.Parameter(torch.tensor(1.), requires_grad=False)

    def get_max_eig(self, coil, mask, dcf=True):
        r""" compute maximal eigenvalue

        References:
            * Beck A, Teboulle M.
              A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
              SIAM J Imaging Sci (2009). DOI: https://doi.org/10.1137/080716542
            * Tan Z, Hohage T, Kalentev O, Joseph AA, Wang X, Voit D, Merboldt KD, Frahm J.
              An eigenvalue approach for the automatic scaling of unknowns in model-based reconstructions: Application to real-time phase-contrast flow MRI.
              NMR Biomed (2017). DOI: https://doi.org/10.1002/nbm.3835
        """
        A = mri.SenseOp(coil, mask, dcf=True)

        x = torch.randn(size=A.ishape, dtype=coil.dtype)
        for _ in range(30):
            y = A.adj(A.fwd(x))
            max_eig = torch.linalg.norm(y).ravel()
            x = y / max_eig

            # print(max_eig)

        return max_eig

    def forward(self,
                curr_x: torch.Tensor,
                x0: torch.Tensor,
                coil: torch.Tensor,
                mask: torch.Tensor,
                max_eig: torch.Tensor) -> torch.Tensor:

        A = mri.SenseOp(coil, mask, dcf=True,
                        device=coil.device)

        grad = A.adj(A.fwd(curr_x) - x0)

        next_x = curr_x - (self.lam / max_eig) * grad

        return next_x

# %%
class VarNet(nn.Module):
    def __init__(self, n_layers, k_iters) -> None:

        super().__init__()

        self.n_cascades = k_iters
        self.dc = data_consistency()
        self.dw = nn.ModuleList([unet2d.Unet(44, 44, num_pool_layers=n_layers).float() for _ in range(k_iters)])

    def forward(self, x0, coil, mask):

        x0 = r2c(x0, axis=1)

        if x0.shape[0] == 1:
            x0 = x0.squeeze(0)

        x  = torch.zeros(size=[mask.shape[-4]] + [int(mask.shape[-2]//2)] * 2, dtype=coil.dtype).to(x0)

        max_eig = self.dc.get_max_eig(coil, mask).to(x0)

        for c in range(self.n_cascades):

            x = self.dc(x, x0, coil, mask, max_eig)

            if x.ndim == 3:
                x = x.unsqueeze(0)

            xr = c2r(x, axis=1)

            print('> xr shape: ', xr.shape)

            N_batch, N_channel, N_frame, N_y, N_x = xr.shape

            # for unet2d
            x3 = torch.reshape(xr, (N_batch, N_channel * N_frame, N_y, N_x))

            xo = self.dw[c](x3.float())

            x5 = torch.reshape(xo, xr.shape)

            # for unet1d
            # x3 = torch.permute(xr, (0, 3, 4, 1, 2))
            # x3 = torch.reshape(x3, (-1, N_channel, N_frame))

            # xo = self.dw[c](x3.float())

            # x5 = torch.reshape(xo, (N_batch, N_y, N_x, N_channel, N_frame))
            # x5 = torch.permute(x5, (0, 3, 4, 1, 2))

            x = x - r2c(x5, axis=1)

        return c2r(x, axis=1)
