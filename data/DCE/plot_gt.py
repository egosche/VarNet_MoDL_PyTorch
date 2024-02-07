"""
This script plots the ground truth

Author:
    Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import h5py
import os

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
fig, ax = plt.subplots(M, N, figsize=(N*fig_height, M*fig_width))
props = dict(boxstyle='round', facecolor='cyan', alpha=0.3)

for d in range(len(data_dirs)):

    dat = data_dirs[d]

    m = int(d / N)
    n = int(d % N)

    print('ind %3d <-> (%3d, %3d)' % (d, m, n))

    f = sio.loadmat(DIR + '/' + dat + '/cart_images.mat')
    cart_image = np.transpose(f['simImg'])[0]
    cart_image = np.swapaxes(cart_image, 0, 1)
    N_y, N_x = cart_image.shape

    ax[m][n].imshow(abs(cart_image), cmap='gray',
                    interpolation=None)
    ax[m][n].set_axis_off()
    ax[m][n].text(0.02 * N_x, 0.08 * N_y, dat,
                  bbox=props, color='w', fontsize=24)

    if d >= 18:
        rect = patches.Rectangle((0, 0), N_y, N_x,
                                 linewidth=4, edgecolor='r',
                                 facecolor='none')

        ax[m][n].add_patch(rect)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(DIR + '/gt.png',
            bbox_inches='tight', pad_inches=0, dpi=300)
