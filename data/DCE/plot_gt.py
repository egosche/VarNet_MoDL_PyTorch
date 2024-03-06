"""
This script plots the ground truth

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import h5py
import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.io as sio

DIR = os.path.dirname(os.path.abspath(__file__))

print(DIR)

# %%
def find_two_ints(input : int):

    N = int(np.ceil(np.sqrt(input)))

    while True:

        if input % N == 0:
            M = int(input / N)
            print(str(input) + ' = ', str(M) + ' x ' + str(N))
            return M, N
        else:
            N = N + 1

# %%
data_dirs = sorted(next(os.walk('.'))[1])

print('available data: ', data_dirs)

M, N = find_two_ints(len(data_dirs))

# %%
fig_width = 4
fig_height = 4

origs = []

for d in range(len(data_dirs)):

    dat = data_dirs[d]

    f = sio.loadmat(DIR + '/' + dat + '/cart_images.mat')
    cart_image = np.transpose(f['simImg'])
    cart_image = np.swapaxes(cart_image, -2, -1)
    origs.append(cart_image)

origs = np.array(origs)


for f in range(origs.shape[1]):

    print('> frame ' + str(f).zfill(2))

    fig, ax = plt.subplots(M, N, figsize=(N*fig_height, M*fig_width))
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.3)

    for v in range(origs.shape[0]):

        vmax = np.amax(abs(origs[v])) * 0.8

        N_y, N_x = origs.shape[-2:]

        m = int(v / N)
        n = int(v % N)

        ax[m][n].imshow(abs(origs[v, f, ...]), cmap='gray',
                        interpolation=None, vmin=0, vmax=vmax)
        ax[m][n].set_axis_off()

        if v >= 18:
            rect = patches.Rectangle((0, 0), N_y, N_x,
                                    linewidth=4, edgecolor='r',
                                    facecolor='none')

            ax[m][n].add_patch(rect)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(DIR + '/frame_' + str(f).zfill(2) + '.png',
                bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
