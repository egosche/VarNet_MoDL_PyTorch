import scipy.io
import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np

from configs.definitions import ROOT_DIR
from datasets.gen_DRO import gen_DRO
from utils import c2r
from models import mri


class modl_dataset(Dataset):
    def __init__(self, mode, num_train, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.mode = mode
        # 'trn' if mode == 'train' else 'tst'
        self.num_train = num_train
        self.dataset_path = dataset_path
        self.sigma = sigma
        self.acquisition_data = scipy.io.loadmat(
            ROOT_DIR / 'data' / 'DCE' / 'Breast_Data_partial.mat',
            squeeze_me=True,
        )['new_data']

    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        # Skip the first samples which are reserved for training
        if self.mode == 'test':
            index = index + self.num_train
        raw_sample = self.acquisition_data[index]

        # Generate simulated DCE
        simImg, smap, _, _, _, _, _, _, _ = gen_DRO(raw_sample)

        traj = None
        traj_path = ROOT_DIR / 'data' / 'DCE' / 'traj.h5'
        with h5.File(traj_path, 'r') as f:
            traj = f['traj'][:]

        gt = np.transpose(simImg)
        csm = np.transpose(smap)
        mask = traj

        x0 = undersample_(gt, csm, mask, self.sigma)

        return (
            torch.from_numpy(c2r(x0)),
            torch.from_numpy(c2r(gt)),
            torch.from_numpy(csm),
            torch.from_numpy(mask),
        )

    def __len__(self):
        if self.mode == 'train':
            return self.num_train
        else:
            return len(self.acquisition_data) - self.num_train


def undersample_(gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape  # this assumes batch size of 1
    csm = csm[None, ...]  # 4dim

    SenseOp = mri.SenseOp(csm, mask, dcf=True, verbose=False)

    b = SenseOp.fwd(gt).detach().cpu().numpy()

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.0**0.5)

    # add noise to k-space data
    # atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return b


def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten() != 0)[0]
    noise = np.random.randn(len(sample_idx) * ncoil) + 1j * np.random.randn(
        len(sample_idx) * ncoil
    )
    noise = noise * (sigma / np.sqrt(2.0))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise  # forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb


def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm  # split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho')  # fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask != 0]
    return k_u


def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask != 0] = b  # zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho')  # ifft
    coil_combine = np.sum(img * csm.conj(), axis=0).astype(
        np.complex64
    )  # coil combine
    return coil_combine
