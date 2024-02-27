import argparse
import h5py
import os

from pathlib import Path

import numpy as np
import scipy.io as sio

DIR = os.path.dirname(os.path.abspath(__file__))

print('> file dir: ', DIR)

# %%
parser = argparse.ArgumentParser(description='prepare dataset for modl training and testing.')

parser.add_argument('--N_training', type=int, default=17,
                    help='number of training data.')

args = parser.parse_args()

# %% read in trajectory
f = h5py.File(DIR + '/traj.h5', 'r')
traj = f['traj'][:]
f.close()

# %%
p = Path(DIR)

SUB_DIR = [d for d in os.listdir(DIR) if os.path.isdir(os.path.join(DIR, d))]
SUB_DIR.sort()

print('> data dir: ', SUB_DIR)


trnCsm = []
trnMask = []
trnOrg = []

tstCsm = []
tstMask = []
tstOrg = []

for n in range(len(SUB_DIR)):

    print('  ', SUB_DIR[n], end="")

    # read in coils
    f = sio.loadmat(SUB_DIR[n] + '/coil_sens.mat')
    coil_sens = np.transpose(f['smap'])

    # read in images
    f = sio.loadmat(SUB_DIR[n] + '/cart_images.mat')
    cart_images = np.transpose(f['simImg'])
    # cart_images = cart_images / np.linalg.norm(cart_images)

    if n < args.N_training:

        print(' - training')

        trnCsm.append(coil_sens)
        trnMask.append(traj)
        trnOrg.append(cart_images)

    else:

        print(' - testing')

        tstCsm.append(coil_sens)
        tstMask.append(traj)
        tstOrg.append(cart_images)


f = h5py.File(DIR + '/dataset.hdf5', 'w')
f.create_dataset('trnCsm', data=np.array(trnCsm))
f.create_dataset('trnMask', data=np.array(trnMask))
f.create_dataset('trnOrg', data=np.array(trnOrg))

f.create_dataset('tstCsm', data=np.array(tstCsm))
f.create_dataset('tstMask', data=np.array(tstMask))
f.create_dataset('tstOrg', data=np.array(tstOrg))

f.close()