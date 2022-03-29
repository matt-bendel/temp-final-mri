import random
import os
import torch
import pytorch_msssim

import numpy as np
import torch.autograd as autograd
import matplotlib.pyplot as plt

from typing import Optional
from data import transforms
from utils.math import complex_abs
from utils.parse_args import create_arg_parser
from wrappers.our_gen_wrapper import get_gan, save_model
from data_loaders.prepare_data import create_data_loaders
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

GLOBAL_LOSS_DICT = {
    'g_loss': [],
    'd_loss': [],
    'mSSIM': [],
    'd_acc': []
}


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


def mssim_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    # ssim_loss = pytorch_ssim.SSIM()
    return pytorch_msssim.msssim(gt, pred)


def compute_gradient_penalty(D, real_samples, fake_samples, args, y):
    """Calculates the gradient penalty loss for WGAN GP"""
    Tensor = torch.FloatTensor
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(args.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(input=interpolates, y=y)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0).to(args.device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    args.in_chans = 32
    args.out_chans = 32

    G, D, opt_G, opt_D, best_loss, start_epoch = get_gan(args)

    train_loader, dev_loader = create_data_loaders(args, big_test=False)

    for epoch in range(start_epoch, args.num_epochs):
        batch_loss = {
            'g_loss': [],
            'd_loss': [],
        }

        for i, data in enumerate(train_loader):
            G.update_gen_status(val=False)
            y, x, y_true, mean, std = data
            y = y.to(args.device)
            x = x.to(args.device)
            y_true = y_true.to(args.device)

            for j in range(args.num_iters_discriminator):
                for param in D.parameters():
                    param.grad = None

                x_hat = G(y, y_true)

                real_pred = D(input=x, y=y)
                fake_pred = D(input=x_hat, y=y)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(D, x.data, x_hat.data, args, y.data)

                d_loss = torch.mean(fake_pred) - torch.mean(
                    real_pred) + args.gp_weight * gradient_penalty + 0.001 * torch.mean(real_pred ** 2)

                d_loss.backward()
                opt_D.step()

            for param in G.gen.parameters():
                param.grad = None

            gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, args.im_size, args.im_size),
                               device=args.device)
            for z in range(args.num_z):
                gens[:, z, :, :, :] = G(y, y_true)

            fake_pred = torch.zeros(size=(y.shape[0], args.num_z), device=args.device)
            for k in range(y.shape[0]):
                cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4])
                cond[0, :, :, :] = y[k, :, :, :]
                cond = cond.repeat(args.num_z, 1, 1, 1)
                temp = D(input=gens[k], y=cond)
                fake_pred[k] = temp[:, 0]

            avg_recon = torch.mean(gens, dim=1)

            gen_pred_loss = torch.mean(fake_pred[0])
            for k in range(y.shape[0] - 1):
                gen_pred_loss += torch.mean(fake_pred[k + 1])

            var_loss = torch.mean(torch.var(gens, dim=1), dim=(0, 1, 2, 3))

            g_loss = -args.adv_weight * torch.mean(gen_pred_loss)
            g_loss += (1 - args.ssim_weight) * F.l1_loss(x, avg_recon) - args.ssim_weight * mssim_tensor(x, avg_recon)
            g_loss += -args.var_weight * var_loss

            g_loss.backward()
            opt_G.step()

            batch_loss['g_loss'].append(g_loss.item())
            batch_loss['d_loss'].append(d_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch + 1, args.num_epochs, i, len(train_loader.dataset) / args.batch_size, d_loss.item(),
                   g_loss.item())
            )

        losses = {
            'psnr': [],
            'ssim': []
        }

        for i, data in enumerate(dev_loader):
            G.update_gen_status(val=True)
            with torch.no_grad():
                y, x, y_true, mean, std = data
                y = y.to(args.device)
                x = x.to(args.device)
                y_true = y_true.to(args.device)

                gens = torch.zeros(size=(y.size(0), args.num_z, args.in_chans, args.im_size, args.im_size),
                                   device=args.device)
                for z in range(args.num_z):
                    gens[:, z, :, :, :] = G(y, y_true)

                avg = torch.mean(gens, dim=1)

                avg_gen = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
                avg_gen[:, :, :, :, 0] = avg[:, 0:16, :, :]
                avg_gen[:, :, :, :, 1] = avg[:, 16:32, :, :]

                gt = torch.zeros(size=(y.size(0), 16, args.im_size, args.im_size, 2), device=args.device)
                gt[:, :, :, :, 0] = x[:, 0:16, :, :]
                gt[:, :, :, :, 1] = x[:, 16:32, :, :]

                for j in range(y.size(0)):
                    avg_gen_np = transforms.root_sum_of_squares(complex_abs(avg_gen[j] * std[j] + mean[j])).cpu().numpy()
                    gt_np = transforms.root_sum_of_squares(complex_abs(gt[j] * std[j] + mean[j])).cpu().numpy()

                    losses['ssim'].append(ssim(gt_np, avg_gen_np))
                    losses['psnr'].append(psnr(gt_np, avg_gen_np))

                    if i == 0 and j == 2:
                        output = transforms.root_sum_of_squares(complex_abs(avg_gen[0] * std[0] + mean[0]))

                        gen_im_list = []
                        for z in range(args.num_z):
                            val_rss = torch.zeros(16, args.im_size, args.im_size, 2).to(args.device)
                            val_rss[:, :, :, 0] = gens[0, z, 0:16, :, :]
                            val_rss[:, :, :, 1] = gens[0, z, 16:32, :, :]
                            gen_im_list.append(transforms.root_sum_of_squares(
                                complex_abs(val_rss * std[0] + mean[0])).cpu().numpy())

                        std_dev = np.zeros(output.shape)
                        for val in gen_im_list:
                            std_dev = std_dev + np.power((val - output), 2)

                        std_dev = std_dev / args.num_z
                        std_dev = np.sqrt(std_dev)

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        im = ax.imshow(std_dev, cmap='viridis')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.subplots_adjust(right=0.85)  # Make room for colorbar

                        # Get position of final error map axis
                        [[x10, y10], [x11, y11]] = ax.get_position().get_points()

                        pad = 0.01
                        width = 0.02
                        cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

                        fig.colorbar(im, cax=cbar_ax)

                        plt.savefig('std_dev_gen.png')
                        plt.close()

        psnr_loss = np.mean(losses['psnr'])
        best_model = psnr_loss > best_loss
        best_loss = psnr_loss if psnr_loss > best_loss else best_loss

        GLOBAL_LOSS_DICT['g_loss'].append(np.mean(batch_loss['g_loss']))
        GLOBAL_LOSS_DICT['d_loss'].append(np.mean(batch_loss['d_loss']))
        GLOBAL_LOSS_DICT['d_acc'].append(np.mean(batch_loss['d_acc']))

        save_str = f"END OF EPOCH {epoch + 1}: [Average D loss: {GLOBAL_LOSS_DICT['d_loss'][epoch - start_epoch]:.4f}] [Average D Acc: {GLOBAL_LOSS_DICT['d_acc'][epoch - start_epoch]:.4f}] [Average G loss: {GLOBAL_LOSS_DICT['g_loss'][epoch - start_epoch]:.4f}]\n"
        print(save_str)
        save_str_2 = f"[Avg PSNR: {np.mean(losses['psnr']):.2f}] [Avg SSIM: {np.mean(losses['ssim']):.4f}]"
        print(save_str_2)

        save_model(args, epoch, G.gen, opt_G, best_loss, best_model, 'generator')
        save_model(args, epoch, D, opt_D, best_loss, best_model, 'discriminator')


if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    torch.backends.cudnn.benchmark = True

    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    train(args)
