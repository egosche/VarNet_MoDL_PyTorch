"""
This module implements MRI operators

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np

import torch
import torch.nn as nn
import torchkbnufft as tkbn
from typing import Optional, Tuple


def fftc(
    input: torch.Tensor | np.ndarray,
    axes: Optional[Tuple] = (-2, -1),
    norm: Optional[str] = 'ortho',
):

    if isinstance(input, np.ndarray):
        tmp = np.fft.ifftshift(input, axes=axes)
        tmp = np.fft.fftn(tmp, axes=axes, norm=norm)
        output = np.fft.fftshift(tmp, axes=axes)

    elif isinstance(input, torch.Tensor):
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

    return output


def ifftc(
    input: torch.Tensor | np.ndarray,
    axes: Optional[Tuple] = (-2, -1),
    norm: Optional[str] = 'ortho',
):

    if isinstance(input, np.ndarray):
        tmp = np.fft.ifftshift(input, axes=axes)
        tmp = np.fft.ifftn(tmp, axes=axes, norm=norm)
        output = np.fft.fftshift(tmp, axes=axes)

    elif isinstance(input, torch.Tensor):
        tmp = torch.fft.ifftshift(input, dim=axes)
        tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
        output = torch.fft.fftshift(tmp, dim=axes)

    return output


class SenseOp:
    """
    Sensitivity Encoding (SENSE) Operators

    Reference:
        * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
          Advances in sensitivity encoding with arbitrary k-space trajectories.
          Magn Reson Med (2001).
    """

    def __init__(
        self,
        coil: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        dcf: Optional[bool] = False,
        verbose=False,
        normalization=False,
        device=torch.device('cpu'),
    ):
        """
        Args:
            coil: [1, N_coil, N_y, N_x]
            mask: [N_frames, N_spokes, N_samples, 2]  # radial trajectory
        """

        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)

        coil = coil.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(device)

        self.coil = coil
        self.mask = mask
        self.dcf = dcf
        self.verbose = verbose
        self.normalization = normalization
        self.device = device

        N_batch = coil.shape[0]
        assert 1 == N_batch

        assert 2 == mask.shape[-1]

        self.N_samples = mask.shape[-2]
        self.N_spokes = mask.shape[-3]
        self.N_frames = mask.shape[-4]

        base_res = int(self.N_samples // 2)

        im_size = [base_res] * 2
        grid_size = [self.N_samples] * 2

        self.ishape = [self.N_frames] + im_size

        self.im_size = im_size
        print(f'im_size: {im_size}\t grid_size: {grid_size}')
        self.NUFFT_FWD = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
        self.NUFFT_ADJ = tkbn.KbNufftAdjoint(
            im_size=im_size, grid_size=grid_size
        )

        self.NUFFT_FWD = self.NUFFT_FWD.to(coil.device)
        self.NUFFT_ADJ = self.NUFFT_ADJ.to(coil.device)

    def _get_normalization_scale(self, nrm_0, nrm_1, output_dim):

        if self.normalization:
            if torch.all(nrm_1 == 0):
                nrm_1 = nrm_1 + 1e-6
            scale = nrm_0 / nrm_1
            for _ in range(output_dim - 1):
                scale = scale.unsqueeze(1)
        else:
            scale = 1

        return scale

    def fwd(self, input) -> torch.Tensor:
        """
        SENSS Forward Operator: from image to k-space
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        input = input.to(self.device)

        if torch.is_floating_point(input):
            input = input + 1j * torch.zeros_like(input)

        N_batch, N_coil, N_y, N_x = self.coil.shape

        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()

        output = []

        for t in range(self.N_frames):

            traj_t = torch.reshape(
                self.mask[..., t, :, :, :], (-1, 2)
            ).transpose(1, 0)
            imag_t = (
                torch.squeeze(input[..., t, :, :]).unsqueeze(0).unsqueeze(0)
            )

            print(
                f'imag_t: {imag_t.shape}\t traj_t: {traj_t.shape}\t coil: {self.coil.shape}'
            )
            grid_t = self.NUFFT_FWD(imag_t, traj_t, smaps=self.coil)

            if self.verbose:
                print('> frame ', str(t).zfill(2))
                print('  traj shape: ', traj_t.shape)
                print('  imag shape: ', imag_t.shape)
                print('  grid shape: ', grid_t.shape)

            output.append(grid_t.detach().cpu().numpy())

        output = torch.tensor(np.array(output)).to(self.coil)
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()

        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output

    def adj(self, input) -> torch.Tensor:
        """
        SENSE Adjoint Operator: from k-space to image
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        input = input.to(self.device)

        if torch.is_floating_point(input):
            input = input + 1j * torch.zeros_like(input)

        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()

        output = []

        for t in range(self.N_frames):

            traj_t = torch.reshape(
                self.mask[..., t, :, :, :], (-1, 2)
            ).transpose(1, 0)

            grid_t = input[t]

            # density compensation function
            if self.dcf:
                comp_t = (
                    traj_t[0, ...] ** 2 + traj_t[1, ...] ** 2
                ) ** 0.5 + 1e-5
                comp_t = comp_t.reshape(1, -1)
                # comp_t = tkbn.calc_density_compensation_function(ktraj=traj_t, im_size=self.im_size)
            else:
                comp_t = 1.0

            imag_t = self.NUFFT_ADJ(grid_t * comp_t, traj_t, smaps=self.coil)

            if self.verbose:
                print('> frame ', str(t).zfill(2))
                print('  traj shape: ', traj_t.shape)
                print('  grid shape: ', grid_t.shape)
                print('  grid shape: ', imag_t.shape)

            output.append(imag_t.detach().cpu().numpy().squeeze())

        output = torch.tensor(np.array(output)).to(self.coil)
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()

        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output


class SenseSp:
    """
    Implementation of the SENSE Operator based on SigPy.
    """
