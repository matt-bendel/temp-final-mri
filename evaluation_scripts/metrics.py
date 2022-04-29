import torch

import numpy as np
import sigpy as sp
from evaluation_scripts import compute_cfid
import matplotlib.pyplot as plt

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

def get_metrics(args):
    G = load_best_gan(args)
    G.update_gen_status(val=True)

    losses = {
        'psnr': [],
        'snr': [],
        'ssim': [],
        'apsd': []
    }

    test_loader = create_test_loader(args)

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            y, x, y_true, mean, std, maps = data
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
                temp_gens[:, z, :, :, :] = gens[:, z, :, :, :] * std[:, None, None, None] + mean[:, None, None, None]

            losses['apsd'].append(torch.mean(torch.std(temp_gens, dim=1), dim=(0, 1, 2, 3)).cpu().numpy())

            avg_gen = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
            avg_gen[:, :, :, :, 0] = avg[:, 0:16, :, :]
            avg_gen[:, :, :, :, 1] = avg[:, 16:32, :, :]

            gt = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
            gt[:, :, :, :, 0] = x[:, 0:16, :, :]
            gt[:, :, :, :, 1] = x[:, 16:32, :, :]

            for j in range(y.size(0)):
                gt_ksp, avg_ksp = tensor_to_complex_np(fft2c_new(gt[j] * std[j] + mean[j]).cpu()), tensor_to_complex_np(fft2c_new(avg_gen[j] * std[j] + mean[j]).cpu())
                avg_gen_np = \
                torch.tensor(get_mvue(avg_ksp.reshape((1,) + avg_ksp.shape), maps[j].reshape((1,) + maps[j].shape)))[0].abs().numpy()
                gt_np = torch.tensor(get_mvue(gt_ksp.reshape((1,) + gt_ksp.shape), maps[j].reshape((1,) + maps[j].shape)))[0].abs().numpy()

                avg_gen_np[np.isnan(avg_gen_np)] = 0
                gt_np[np.isnan(gt_np)] = 0

                losses['ssim'].append(ssim(gt_np, avg_gen_np))
                losses['psnr'].append(psnr(gt_np, avg_gen_np))
                losses['snr'].append(snr(gt_np, avg_gen_np))

    print(f'MEAN PSNR: {np.mean(losses["psnr"]):.2f} || MEDIAN PSNR: {np.median(losses["psnr"]):.2f}')
    print(f'MEAN SNR: {np.mean(losses["snr"]):.2f} || MEDIAN SNR: {np.median(losses["snr"]):.2f}')
    print(f'MEAN SSIM: {np.mean(losses["ssim"]):.4f} || MEDIAN SSIM: {np.median(losses["ssim"]):.4f}')
    print(f'MEAN APSD: {np.mean(losses["apsd"]):.2f} || MEDIAN APSD: {np.median(losses["apsd"]):.2f}')

    compute_cfid.get_cfid(args, G)

