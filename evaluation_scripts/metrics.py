import torch
import os
import numpy as np
import sigpy as sp
from evaluation_scripts import compute_cfid
import matplotlib.pyplot as plt
import imageio as iio

from typing import Optional
from wrappers.our_gen_wrapper import load_best_gan
from data_loaders.prepare_data import create_test_loader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.math import tensor_to_complex_np
from utils.fftc import ifft2c_new, fft2c_new
from data import transforms
from utils.math import complex_abs


def psnr(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)

    return psnr_val


def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)
    snr = 10 * np.log10(np.mean(gt ** 2) / noise_mse)

    return snr


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred) ** 2)

    return noise_mse


def ssim(
        gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    # if not gt.ndim == 3:
    #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
        gt, pred, data_range=maxval
    )

    return ssim


def get_mvue(kspace, s_maps):
    return np.sum(sp.ifft(kspace, axes=(-1, -2)) * np.conj(s_maps), axis=1) / np.sqrt(
        np.sum(np.square(np.abs(s_maps)), axis=1))


def generate_image(fig, target, image, method, image_ind, rows, cols, kspace=False, disc_num=False):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT' and method != 'Std. Dev':
        psnr_val = psnr(target, image)
        snr_val = snr(target, image)
        ssim_val = ssim(target, image)
        if not kspace:
            pred = disc_num
            ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}, Pred: {pred * 100:.2f}% True') if disc_num else ax.set_title(
                f'PSNR: {psnr_val:.2f}, SNR: {snr_val:.2f}\nSSIM: {ssim_val:.4f}')

    if method == 'Std. Dev':
        im = ax.imshow(image, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if kspace:
            image = image ** 0.4
            target = target ** 0.4
        im = ax.imshow(np.abs(image), cmap='gray', vmin=0, vmax=np.max(target))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(method)

    return im, ax


def generate_error_map(fig, target, recon, method, image_ind, rows, cols, relative=False, k=1, kspace=False):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)  # Add to subplot

    # Normalize error between target and reconstruction
    if kspace:
        recon = recon ** 0.4
        target = target ** 0.4

    error = (target - recon) if relative else np.abs(target - recon)
    # normalized_error = error / error.max() if not relative else error
    if relative:
        im = ax.imshow(k * error, cmap='bwr', origin='lower', vmin=-0.0001, vmax=0.0001)  # Plot image
        plt.gca().invert_yaxis()
    else:
        im = ax.imshow(k * error, cmap='jet', vmax=1) if kspace else ax.imshow(k * error, cmap='jet', vmax=0.0001)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Return plotted image and its axis in the subplot
    return im, ax


def gif_im(true, gen_im, lang_true, lang_im, index, type, disc_num=False):
    fig = plt.figure()

    generate_image(fig, true, gen_im, f'z {index}', 1, 2, 2, disc_num=False)
    generate_image(fig, lang_true, lang_im, f'z {index}', 2, 2, 2, disc_num=False)
    im, ax = generate_error_map(fig, true, gen_im, f'z {index}', 3, 2, 2)
    im, ax = generate_error_map(fig, lang_true, lang_im, f'z {index}', 4, 2, 2)

    plt.savefig(f'/home/bendel.8/Git_Repos/temp-final-mri/gif_{type}_{index - 1}.png')
    plt.close()


def generate_gif(type):
    images = []
    for i in range(8):
        images.append(iio.imread(f'/home/bendel.8/Git_Repos/temp-final-mri/gif_{type}_{i}.png'))

    iio.mimsave(f'variation_gif.gif', images, duration=0.25)

    for i in range(8):
        os.remove(f'/home/bendel.8/Git_Repos/temp-final-mri/gif_{type}_{i}.png')


def get_colorbar(fig, im, ax, left=False):
    fig.subplots_adjust(right=0.85)  # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.01
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10]) if not left else fig.add_axes(
        [x10 - 2 * pad, y10, width, y11 - y10])

    cbar = fig.colorbar(im, cax=cbar_ax, format='%.2e')  # Generate colorbar
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.locator_params(nbins=5)

    if left:
        cbar_ax.yaxis.tick_left()
        cbar_ax.yaxis.set_label_position('left')


def get_plots(fname, gt_np, avg_gen_np, temp_gens, R, slice, maps, ind):
    recons = np.zeros((32, 384, 384))
    recon_object = None
    gen_recons = np.zeros((32, 384, 384))
    recon_directory = f'/storage/fastMRI_brain/Langevin_Recons_R={R}/'

    for j in range(32):
        try:
            new_filename = recon_directory + fname + f'|langevin|slide_idx_{slice}_R={R}_sample={j}_outputs.pt'
            recon_object = torch.load(new_filename)
        except:
            print('EXCEPT')
            return

        recons[j] = complex_abs(recon_object['mvue'][0].permute(1, 2, 0)).cpu().numpy()
        gt_lang = recon_object['gt'][0][0].abs().cpu().numpy()

        ksp = tensor_to_complex_np(fft2c_new(temp_gens[j]).cpu())
        gen_recons[j] = \
            torch.tensor(
                get_mvue(ksp.reshape((1,) + ksp.shape), maps.reshape((1,) + maps.shape)))[
                0].abs().numpy()

        gif_im(gt_np, gen_recons[j], gt_lang, recons[j], j, 'image')

    generate_gif('image')

    avg_lang = np.mean(recons, axis=0)
    gt_lang = recon_object['gt'][0][0].abs().cpu().numpy()
    zfr_lang = recon_object['zfr'][0].cpu().numpy()

    fig = plt.figure()
    fig.subplots_adjust(wspace=0, hspace=0.05)

    generate_image(fig, gt_lang, gt_lang, f'GT', 1, 2, 4, disc_num=False)
    generate_image(fig, gt_lang, zfr_lang, f'ZFR', 2, 2, 4, disc_num=False)
    generate_image(fig, gt_lang, avg_lang, f'CSGM', 3, 2, 4, disc_num=False)
    generate_image(fig, gt_np, avg_gen_np, f'RC-GAN', 4, 2, 4, disc_num=False)

    im, ax = generate_error_map(fig, gt_lang, zfr_lang, f'ZFR', 6, 2, 4)
    generate_error_map(fig, gt_lang, avg_lang, f'CSGM', 6, 2, 4)
    generate_error_map(fig, gt_np, avg_gen_np, f'RC-GAN', 6, 2, 4)

    get_colorbar(fig, im, ax, left=True)

    plt.savefig(f'comp_plots_{ind}_{0}.png')
    plt.close(fig)


def get_metrics(args):
    G = load_best_gan(args)
    G.update_gen_status(val=True)

    losses = {
        'psnr': [],
        'snr': [],
        'ssim': [],
        'apsd': [],
        'mse': [],
        'max_i': []
    }

    test_loader = create_test_loader(args)

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            y, x, y_true, mean, std, maps, fname, slice = data
            y = y.to(args.device)
            x = x.to(args.device)
            y_true = y_true.to(args.device)
            maps = maps.cpu().numpy()

            gens = torch.zeros(size=(y.size(0), 32, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            for z in range(32):
                gens[:, z, :, :, :] = G(y, y_true)

            avg = torch.mean(gens, dim=1)

            temp_gens = torch.zeros(gens.shape, dtype=gens.dtype)
            for z in range(32):
                temp_gens[:, z, :, :, :] = gens[:, z, :, :, :] * std[:, None, None, None].to(args.device) + mean[:,
                                                                                                            None, None,
                                                                                                            None].to(
                    args.device)

            losses['apsd'].append(torch.mean(torch.std(temp_gens, dim=1), dim=(0, 1, 2, 3)).cpu().numpy())

            avg_gen = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
            avg_gen[:, :, :, :, 0] = avg[:, 0:16, :, :]
            avg_gen[:, :, :, :, 1] = avg[:, 16:32, :, :]

            gt = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
            gt[:, :, :, :, 0] = x[:, 0:16, :, :]
            gt[:, :, :, :, 1] = x[:, 16:32, :, :]

            for j in range(y.size(0)):
                gt_ksp, avg_ksp = tensor_to_complex_np(fft2c_new(gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(
                    fft2c_new(avg_gen[j] * std[j] + mean[j]).cpu())
                avg_gen_np = \
                    torch.tensor(
                        get_mvue(avg_ksp.reshape((1,) + avg_ksp.shape), maps[j].reshape((1,) + maps[j].shape)))[
                        0].abs().numpy()
                gt_np = \
                    torch.tensor(get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps[j].reshape((1,) + maps[j].shape)))[
                        0].abs().numpy()

                if i % 3 == 0 and j == 0:
                    get_plots(fname[j], gt_np, avg_gen_np, temp_gens[j, :, :, :, :], args.R, slice[j], maps[j])

                avg_gen_np[np.isnan(avg_gen_np)] = 0
                gt_np[np.isnan(gt_np)] = 0

                losses['ssim'].append(ssim(gt_np, avg_gen_np))
                losses['psnr'].append(psnr(gt_np, avg_gen_np))
                losses['snr'].append(snr(gt_np, avg_gen_np))
                losses['mse'].append(mse(gt_np, avg_gen_np))
                losses['max_i'].append(gt_np.max())

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Metric Histograms')
    fig.subplots_adjust(hspace=1)
    ax1.hist(losses['psnr'], bins=15)
    ax1.set_title('PSNR')
    ax2.hist(losses['snr'], bins=15)
    ax2.set_title('SNR')
    plt.savefig('histo.png')
    plt.close(fig)

    fig = plt.figure()
    fig.suptitle('MSE vs. MAX_I')
    plt.scatter(losses['max_i'], losses['mse'])
    plt.xlabel('MAX_I')
    plt.ylabel('MSE')
    plt.savefig('mse_v_maxi.png')
    plt.close(fig)

    print(f'MEAN PSNR: {np.mean(losses["psnr"]):.2f} || MEDIAN PSNR: {np.median(losses["psnr"]):.2f}')
    print(f'MEAN SNR: {np.mean(losses["snr"]):.2f} || MEDIAN SNR: {np.median(losses["snr"]):.2f}')
    print(f'MEAN SSIM: {np.mean(losses["ssim"]):.4f} || MEDIAN SSIM: {np.median(losses["ssim"]):.4f}')
    print(f'MEAN APSD: {np.mean(losses["apsd"]):} || MEDIAN APSD: {np.median(losses["apsd"]):}')

    exit()
    compute_cfid.get_cfid(args, G)
