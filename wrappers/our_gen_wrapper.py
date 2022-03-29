import pathlib
import shutil
import torch
import numpy as np

from utils.fftc import ifft2c_new, fft2c_new
from utils.get_mask import get_mask


# THIS FILE CONTAINTS UTILITY FUNCTIONS FOR OUR GAN AND A WRAPPER CLASS FOR THE GENERATOR
def get_gan(args):
    from utils.prepare_models import build_model, build_optim, build_discriminator

    if args.resume:
        checkpoint_file_gen = pathlib.Path(
            f'{args.checkpoint_dir}/generator_model.pt')
        checkpoint_gen = torch.load(checkpoint_file_gen, map_location=torch.device('cuda'))

        checkpoint_file_dis = pathlib.Path(
            f'{args.checkpoint_dir}/discriminator_model.pt')
        checkpoint_dis = torch.load(checkpoint_file_dis, map_location=torch.device('cuda'))

        generator = build_model(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        generator.load_state_dict(checkpoint_gen['model'])

        generator = GANWrapper(generator, args)

        opt_gen = build_optim(args, generator.gen.parameters())
        opt_gen.load_state_dict(checkpoint_gen['optimizer'])

        discriminator.load_state_dict(checkpoint_dis['model'])

        opt_dis = build_optim(args, discriminator.parameters())
        opt_dis.load_state_dict(checkpoint_dis['optimizer'])

        best_loss = checkpoint_gen['best_dev_loss']
        start_epoch = checkpoint_gen['epoch']

    else:
        generator = build_model(args)
        discriminator = build_discriminator(args)

        if args.data_parallel:
            generator = torch.nn.DataParallel(generator)
            discriminator = torch.nn.DataParallel(discriminator)

        generator = GANWrapper(generator, args)

        # Optimizers
        opt_gen = torch.optim.Adam(generator.gen.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        opt_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        best_loss = 0
        start_epoch = 0

    return generator, discriminator, opt_gen, opt_dis, best_loss, start_epoch


def save_model(args, epoch, model, optimizer, best_dev_loss, is_new_best, m_type):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': args.exp_dir
        },
        f=args.exp_dir / f'standard' / f'{m_type}_model.pt'
    )

    if is_new_best:
        shutil.copyfile(args.exp_dir / f'standard' / f'{m_type}_model.pt',
                        args.exp_dir / f'standard' / f'{m_type}_best_model.pt')


class GANWrapper:
    def __init__(self, gen, args):
        self.args = args
        self.resolution = args.im_size
        self.gen = gen
        self.data_consistency = self.args.data_consistency

    def get_noise(self, num_vectors):
        # return torch.cuda.FloatTensor(np.random.normal(size=(num_vectors, self.args.latent_size), scale=1))
        return torch.rand((num_vectors, 2, self.resolution, self.resolution)).cuda()

    def update_gen_status(self, val):
        self.gen.eval() if val else self.gen.train()

    def reformat(self, samples):
        reformatted_tensor = torch.zeros(size=(samples.size(0), 16, self.resolution, self.resolution, 2),
                                         device=self.args.device)
        reformatted_tensor[:, :, :, :, 0] = samples[:, 0:16, :, :]
        reformatted_tensor[:, :, :, :, 1] = samples[:, 16:32, :, :]

        return reformatted_tensor

    def readd_measures(self, samples, measures):
        reformatted_tensor = self.reformat(samples)
        reconstructed_kspace = fft2c_new(reformatted_tensor)

        inds = get_mask(self.resolution)

        reconstructed_kspace[:, :, inds[0], inds[1], :] = measures[:, :, inds[0], inds[1], :]

        image = ifft2c_new(reconstructed_kspace)

        output_im = torch.zeros(size=samples.shape, device=self.args.device)
        output_im[:, 0:16, :, :] = image[:, :, :, :, 0]
        output_im[:, 16:32, :, :] = image[:, :, :, :, 1]

        return output_im

    def __call__(self, y, true_measures):
        num_vectors = y.size(0)
        z = self.get_noise(num_vectors)
        samples = self.gen(torch.cat([y, z], dim=1))
        samples = self.readd_measures(samples, true_measures)

        return samples
