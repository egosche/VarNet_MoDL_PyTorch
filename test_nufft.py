"""
This script tests NUFFT:

* density compensation function
* normalization strategy
* power iteration
* gradient descent

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

from models import mri

import h5py
import os
import torch

import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
print(DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pathlib
OUT_DIR = DIR + '/test_nufft'
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# %%
f = h5py.File(DIR + '/data/DCE/dataset.hdf5', 'r')

index = [0]

train_orig = f['trnOrg'][index]
train_coil = f['trnCsm'][index]
train_traj = f['trnMask'][index]

print(len(f['trnMask']))
print(len(f['tstMask']))

print('train orig shape:', train_orig.shape)
print('train coil shape:', train_coil.shape)
print('train mask shape:', train_traj.shape)

f.close()

# %%
from datasets import modl_dataset

x0_ = modl_dataset.undersample_(np.squeeze(train_orig),
                                np.squeeze(train_coil),
                                np.squeeze(train_traj), 0.01)

print('> x0_ shape: ', x0_.shape)

# %%
def iter_nufft(dcf=True, normalization=True):

    A = mri.SenseOp(train_coil, train_traj, dcf=dcf,
                    normalization=normalization, device=device)

    b = A.fwd(np.squeeze(train_orig))

    print('> radial spokes shape: ', b.shape)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = (noise * 0.01 / (2.**0.5)).to(device)

    b = b + noise

    x0 = torch.from_numpy(x0_).to(device)
    x = torch.zeros(size=A.ishape, dtype=x0.dtype).to(device)

    # compute maximal eigenvalue:
    z = torch.randn(size=A.ishape, dtype=x0.dtype).to(device)
    for n in range(15):
        y = A.adj(A.fwd(z))
        max_eig = torch.linalg.norm(y).ravel()
        z = y / max_eig

        print(max_eig)

    x_iter = []

    for n in range(10):

        x_old = x.clone()
        grad = A.adj(A.fwd(x) - b)

        x = x - (1./max_eig) * grad

        resid = (torch.linalg.norm(x - x_old).ravel()).ravel()
        print('> iter ' + str(n).zfill(4) + ' residuum ' + str(resid[0]))

        x_iter.append(x.detach().cpu().numpy())

    x_iter = np.array(x_iter)
    return x_iter

# %%
x_dcf0_norm0 = iter_nufft(dcf=False, normalization=False)
x_dcf1_norm0 = iter_nufft(dcf=True , normalization=False)
x_dcf1_norm1 = iter_nufft(dcf=True , normalization=True)

# %%
N_iter, N_time, N_y, N_x = x_dcf0_norm0.shape

for n in range(N_iter):

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    I0 = np.swapaxes(abs(x_dcf0_norm0[n, 0]), 0, 1)
    I1 = np.swapaxes(abs(x_dcf1_norm0[n, 0]), 0, 1)
    I2 = np.swapaxes(abs(x_dcf1_norm1[n, 0]), 0, 1)

    ax[0].imshow(I0, cmap='gray', vmin=0)
    ax[0].set_title('DCF 0 NORM 0')
    ax[1].imshow(I1, cmap='gray', vmin=0)
    ax[1].set_title('DCF 1 NORM 0')
    ax[2].imshow(I2, cmap='gray', vmin=0)
    ax[2].set_title('DCF 1 NORM 1')

    for m in range(3):
        ax[m].set_axis_off()
        ax[m].text(0.03 * N_y, 0.08 * N_x, 'Iter ' + str(n+1).zfill(2),
                   color='w', weight='bold',fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(OUT_DIR + '/nufft_iter' + str(n) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=300)
