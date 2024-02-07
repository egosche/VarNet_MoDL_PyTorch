import argparse
import h5py
import os
import torch

from sigpy import linop
from sigpy.mri import app

import numpy as np
import sigpy as sp
import scipy.io as sio

DIR = os.path.dirname(os.path.abspath(__file__))

device = sp.Device(0 if torch.cuda.is_available() else -1)

# %%
parser = argparse.ArgumentParser(description='run nufft & cs reconstruction.')

parser.add_argument('--src_file', type=str, default='BC29',
                    help='src (source) data directory [default: BC29]')

parser.add_argument('--spokes', type=int, default=21)

parser.add_argument('--sigma', type=float, default=0.01,
                    help='k-space noise standard deviation.')

parser.add_argument('--lamda', type=float, default=0.001,
                    help='regularization.')

args = parser.parse_args()

# %%

def sim_traj(N_time=22, N_spokes=13, base_res=320, golden_idx=2):

    N_tot_spokes = N_spokes * N_time
    N_samples = base_res * 2

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (golden_idx + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    return traj


def sim_kdat(cart_ksp, traj, N_time=22, N_coil=16, N_spokes=13, base_res=320):

    N_samples = base_res * 2
    ishape = cart_ksp.shape[1:]

    radialx_ksp = np.zeros([N_time, N_coil, N_spokes, N_samples], dtype=complex)

    for n in range(0, N_time):
        traj_r = traj[n, ...]
        temp_i = sp.ifft(cart_ksp, axes=(-2, -1))[n, ...]
        F = linop.NUFFT(ishape, traj_r)
        radialx_ksp[n, ...] = F * temp_i

    return radialx_ksp

# %%
DAT_DIR = DIR + '/' + args.src_file
assert os.path.isdir(os.path.join(DAT_DIR))

f = sio.loadmat(DAT_DIR + '/cart_images.mat')
cart_images = np.transpose(f['simImg'])
N_frame, N_y, N_x = cart_images.shape
cart_images = cart_images[:, None, ...]
assert N_y == N_x  # isotropic FOv
print('cart image shape: ', cart_images.shape)

f = sio.loadmat(DAT_DIR + '/coil_sens.mat')
coil_sens = np.transpose(f['smap'])
N_coil = coil_sens.shape[0]
print('coil sens shape: ', coil_sens.shape)

base_res = N_x
N_spokes = args.spokes

orig_ksp = sp.fft(cart_images * coil_sens, axes=(-2, -1))

trj = sim_traj(N_spokes=N_spokes, base_res=base_res)
print('sim trj shape: ', trj.shape)

ksp = sim_kdat(orig_ksp, trj, N_time=N_frame, N_coil=N_coil,
               N_spokes=N_spokes, base_res=base_res)
print('sim ksp shape: ', ksp.shape)

# add noise
noise = torch.randn(ksp.shape) + 1j * torch.randn(ksp.shape)
noise = noise * args.sigma / (2.**0.5)

ksp = ksp + noise.detach().cpu().numpy()

dcf = (trj[..., 0]**2 + trj[..., 1]**2)**0.5

# %% NUFFT recon
R_nufft = []

for f in range(N_frame):
    k1 = ksp[f]
    t1 = trj[f]
    d1 = dcf[f]

    N = linop.NUFFTAdjoint(oshape=(N_coil, N_x, N_x), coord=t1)

    R1 = N(k1 * d1)
    R_nufft.append(np.sum(np.abs(R1)**2, axis=0)**0.5)

R_nufft = np.array(R_nufft)
print(R_nufft.shape)

# %% Temporal TV regularization
ksp_6dim = ksp[:, None, :, None, :, :]
mps_4dim = coil_sens[:, None, ...]

R_tv = app.HighDimensionalRecon(ksp_6dim, mps_4dim,
                                combine_echo=False,
                                lamda=args.lamda,
                                coord=trj,
                                regu='TV', regu_axes=[0],
                                max_iter=10,
                                solver='ADMM', rho=0.1,
                                device=device,
                                show_pbar=False,
                                verbose=True).run()

R_tv = sp.to_device(R_tv)
print(R_tv.shape)

# %%
f = h5py.File(DAT_DIR + '/nufftcs.h5', 'w')
f.create_dataset('fully', data=cart_images)
f.create_dataset('nufft', data=R_nufft)
f.create_dataset('cs', data=R_tv)
f.close()