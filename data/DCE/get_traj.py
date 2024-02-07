import argparse
import h5py
import os
import torch

import numpy as np
import sigpy as sp

DIR = os.path.dirname(os.path.abspath(__file__))

device = sp.Device(0 if torch.cuda.is_available() else -1)

# %%
parser = argparse.ArgumentParser(description='run nufft & cs reconstruction.')

parser.add_argument('--time', type=int, default=22)

parser.add_argument('--spokes', type=int, default=13)

parser.add_argument('--base_res', type=int, default=320)

parser.add_argument('--golden_idx', type=int, default=2)

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

    # traj = traj / 2

    traj = traj.reshape(N_time, N_spokes, N_samples, 2)

    # scale to [-pi, pi]
    traj = traj / base_res * np.pi

    return traj


# %%
traj = sim_traj(args.time, args.spokes, args.base_res, args.golden_idx)

print('> traj shape: ', traj.shape)
print('> traj max: ', np.max(abs(traj)))

f = h5py.File(DIR + '/traj.h5', 'w')
f.create_dataset('traj', data=traj)
f.close()