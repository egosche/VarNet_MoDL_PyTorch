import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sigpy as sp

from sigpy import linop, util
from sigpy.mri import app

def create_radial_traj(N_x, N_time, N_spokes, N_samples):
    N_tot_spokes = N_spokes * N_time
    base_res = N_x

    base_lin = np.arange(N_samples).reshape(1, -1) - base_res

    tau = 0.5 * (1 + 5**0.5)
    base_rad = np.pi / (2 + tau - 1)

    base_rot = np.arange(N_tot_spokes).reshape(-1, 1) * base_rad

    traj = np.zeros((N_tot_spokes, N_samples, 2))
    traj[..., 0] = np.cos(base_rot) @ base_lin
    traj[..., 1] = np.sin(base_rot) @ base_lin

    return traj / 2


def test_nufft(traj, coil_sens, kspace, N_time, N_spokes, N_samples):
    traj_t = traj.reshape(N_time, N_spokes, N_samples, 2)
    traj_r = traj_t[10, ...]

    F = linop.NUFFT(coil_sens.shape, traj_r)
    dcf = (traj_r[..., 0]**2 + traj_r[..., 1]**2)**0.5

    temp_img = sp.ifft(kspace, axes=(-2, -1))[10, ...]  # one image frame
    temp_ksp = F * temp_img
    temp_img_FH = F.H * (temp_ksp * dcf)

    print('F ishape: ' + str(F.ishape))
    print('F oshape: ' + str(F.oshape))

    print(str(temp_img_FH.shape))
    temp_img_FH_rss = np.sum(np.abs(temp_img_FH)**2, axis=0)**0.5

    return traj_t


def undersample(traj_t, coil_sens, kspace, N_time, N_coil, N_spokes, N_samples):
    radialx_ksp = np.zeros([N_time, N_coil, N_spokes, N_samples], dtype=complex)

    for n in range(0, N_time):
        traj_r = traj_t[n, ...]
        temp_i = sp.ifft(kspace, axes=(-2, -1))[n, ...]
        F = linop.NUFFT(coil_sens.shape, traj_r)
        radialx_ksp[n, ...] = F * temp_i

    print(np.max(np.abs(radialx_ksp)))
    print(np.min(np.abs(radialx_ksp)))
    return radialx_ksp

def tv_recon(radialx_ksp_n, traj_t, coil_sens, lamda):
    radialx_ksp6 = radialx_ksp_n[:, None, :, None, :, :]
    orig_mps4 = coil_sens[:, None, :, :]
    print(traj_t.shape)

    radialx_tv = app.HighDimensionalRecon(radialx_ksp6, orig_mps4,
                            combine_echo=False,
                            lamda=lamda,
                            coord=traj_t,
                            regu='TV', regu_axes=[0],
                            max_iter=10,
                            solver='ADMM', rho=0.1,
                            device=sp.Device(0),
                            show_pbar=False,
                            verbose=True).run()

    radialx_tv = sp.to_device(radialx_tv)
    return radialx_tv


def main(args):
    file_path = args.file_path
    N_spokes = args.n_spokes

    # Load test file
    cart_images_mat = scipy.io.loadmat(file_path + 'cart_images.mat')
    cart_images = cart_images_mat['simImg']

    coil_sens_mat = scipy.io.loadmat(file_path + 'coil_sens.mat')
    coil_sens = coil_sens_mat['smap']

    cart_images = cart_images.transpose(2, 0, 1)
    coil_sens = coil_sens.transpose(2, 0, 1)

    cart_images = np.flip(cart_images, axis=2)
    coil_sens = np.flip(coil_sens, axis=2)

    print(f'Cart images shape: {str(cart_images.shape)}')
    print(f'Coil sens shape: {str(coil_sens.shape)}')

    N_time = cart_images.shape[0]
    N_coil = coil_sens.shape[0]

    N_y, N_x = coil_sens.shape[-2:]

    # Get trajectory
    base_res = N_x
    N_samples = base_res * 2
    traj = create_radial_traj(N_x, N_time, N_spokes, N_samples)

    tmp = np.expand_dims(cart_images, axis=1)
    coilimg = tmp * coil_sens
    kspace = sp.fft(coilimg, axes=(-2, -1))

    traj_t = test_nufft(traj, coil_sens, kspace, N_time, N_spokes, N_samples)

    # Radial undersample
    radialx_ksp = undersample(traj_t, coil_sens, kspace, N_time, N_coil, N_spokes, N_samples)

    # Add noise
    noise = 0.001 * util.randn(radialx_ksp.shape, dtype=radialx_ksp.dtype)
    radialx_ksp_n = radialx_ksp + noise

    # Recon using TV
    radialx_tv_p3 = tv_recon(radialx_ksp_n, traj_t, coil_sens, 0.001)
    radialx_tv_p7 = tv_recon(radialx_ksp_n, traj_t, coil_sens, 0.0000001)
    radialx_tv_p1 = tv_recon(radialx_ksp_n, traj_t, coil_sens, 0.1)

    # Save recons
    np.savez('radialx_tv.npz', radialx_tv_p3=radialx_tv_p3, radialx_tv_p7=radialx_tv_p7, radialx_tv_p1=radialx_tv_p1)


        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file_path", type=str, default='./data/DCE/BC31/')
    parser.add_argument("--n_spokes", type=int, default=13)
    args = parser.parse_args()

    main(args)