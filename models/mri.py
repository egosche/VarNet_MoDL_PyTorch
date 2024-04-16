"""
This module implements MRI operators

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import sigpy
import torch
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
        device=torch.device('cuda'),
    ):
        """
        Args:
            coil: [1, N_coil, N_y, N_x]
            mask: [N_frames, N_spokes, N_samples, 2]  # radial trajectory
        """

        if device.type == 'cuda':
            sigpy.Device(0)

        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)

        coil = coil.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(device)

        self.coil = coil  # [1, 16, 320, 320]
        self.mask = mask  # [22, 13, 640, 2] or [1, 22, 13, 640, 2]
        self.dcf = dcf
        self.verbose = verbose
        self.normalization = normalization
        self.device = device

        N_batch = coil.shape[0]
        assert 1 == N_batch
        assert 2 == mask.shape[-1]

        self.N_frames, self.N_spokes, self.N_samples, _ = (
            self.mask.squeeze().shape
        )
        self.ishape = [self.N_frames] + list(
            self.coil.shape[1:]
        )  # [22, 16, 320, 320]

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

        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()

        output = []

        # Create multi-coil multi-frame image data
        if input.shape[1] != self.coil.shape[1]:
            tmp = input.squeeze()[:, None, :, :]
            input = tmp * self.coil  # [22, 16, 320, 320]

        for t in range(self.N_frames):
            traj_t = self.mask.squeeze()[t]
            NUFFT_FWD = sigpy.linop.NUFFT(
                ishape=self.coil.shape[1:], coord=traj_t
            )
            NUFFT_FWD = sigpy.to_pytorch_function(NUFFT_FWD)

            imag_t = input[t].squeeze()

            grid_t = NUFFT_FWD.apply(imag_t)
            grid_t = torch.view_as_complex(grid_t)

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

        return output  # [22, 16, 13, 640]

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
            traj_t = self.mask.squeeze()[t]

            # density compensation function
            if self.dcf:
                # compute distance from 0, 0 coord for all traj coords
                # r = sqrt(y ** 2 + x ** 2)
                dcomp = torch.sqrt(torch.sum(traj_t**2, axis=-1))

                # normalize between 1/N_samples and 1
                margin = dcomp.max() - dcomp.min()
                scaling = (1 - 1 / self.N_samples) / margin
                dcomp = (dcomp - dcomp.min()) * scaling + 1 / self.N_samples

                # add pseudo coil dim
                dcomp = dcomp[None, :]
            else:
                dcomp = 1.0

            NUFFT_ADJ = sigpy.linop.NUFFTAdjoint(
                oshape=self.coil.shape[1:], coord=traj_t
            )
            NUFFT_ADJ = sigpy.to_pytorch_function(NUFFT_ADJ)

            grid_t = input[t]

            imag_t = NUFFT_ADJ.apply(grid_t * dcomp)
            imag_t = torch.view_as_complex(imag_t)

            if self.verbose:
                print('> frame ', str(t).zfill(2))
                print('  traj shape: ', traj_t.shape)
                print('  grid shape: ', grid_t.shape)
                print('  grid shape: ', imag_t.shape)

            output.append(imag_t.detach().cpu().numpy().squeeze())

        output = torch.tensor(np.array(output)).to(self.coil)
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()

        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale  # [22, 16, 320, 320]

        output = torch.sum(output * self.coil.conj(), dim=1, keepdim=False)

        return output


class SenseSp:
    """
    Implementation of the SENSE Operator based on SigPy.
    """
