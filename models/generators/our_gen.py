"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU()
        )
        self.conv_1x1 = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        return self.conv_1x1(x) + self.conv_block(x)


class ConvDownBlock(nn.Module):
    def __init__(self, in_chans, out_chans, batch_norm=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.batch_norm = batch_norm

        self.conv_1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.res = ResidualBlock(out_chans)
        self.conv_3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_chans)
        self.activation = nn.PReLU()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        if self.batch_norm:
            out = self.activation(self.bn(self.conv_1(input)))
            skip_out = self.res(out)
            out = self.conv_3(skip_out)
        else:
            out = self.activation(self.conv_1(input))
            skip_out = self.res(out)
            out = self.conv_3(skip_out)

        return out, skip_out


class ConvUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv_1 = nn.ConvTranspose2d(in_chans // 2, in_chans // 2, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(in_chans // 2)
        self.activation = nn.PReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.PReLU(),
            ResidualBlock(out_chans),
        )

    def forward(self, input, skip_input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        residual_skip = skip_input  # self.res_skip(skip_input)
        upsampled = self.activation(self.bn(self.conv_1(input, output_size=residual_skip.size())))
        concat_tensor = torch.cat([residual_skip, upsampled], dim=1)

        return self.layers(concat_tensor)


class GeneratorModel(nn.Module):
    def __init__(self, in_chans, out_chans, resolution, latent_size=512):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = 64
        self.num_pool_layers = 5
        self.latent_size = latent_size
        self.resolution = resolution

        num_pool_layers = self.num_pool_layers

        ch = self.chans

        self.down_sample_layers = nn.ModuleList([ConvDownBlock(in_chans, ch, batch_norm=False)])
        for i in range(num_pool_layers - 1):
            if i < 3:
                self.down_sample_layers += [ConvDownBlock(ch, ch * 2)]
                ch *= 2
            else:
                self.down_sample_layers += [ConvDownBlock(ch, ch)]

        self.res_layer_1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
            ResidualBlock(ch),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(),
            # nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(ch),
            # nn.PReLU()
        )

        # self.middle_z_grow_conv = nn.Sequential(
        #     nn.Conv2d(latent_size // 4, latent_size // 2, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv2d(latent_size // 2, latent_size, kernel_size=(3, 3), padding=1),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        #
        # if self.resolution == 384:
        #     self.middle_z_grow_linear = nn.Sequential(
        #         nn.Linear(latent_size, latent_size // 4 * 3 * 3),
        #         nn.LeakyReLU(negative_slope=0.2),
        #         nn.Linear(latent_size // 4 * 3 * 3, latent_size // 4 * 6 * 6),
        #         nn.LeakyReLU(negative_slope=0.2),
        #         nn.Linear(latent_size // 4 * 6 * 6, latent_size // 4 * 12 * 12),
        #         nn.LeakyReLU(negative_slope=0.2),
        #     )
        # else:
        #     self.middle_z_grow_linear = nn.Sequential(
        #         nn.Linear(latent_size, latent_size // 4 * 2 * 2),
        #         nn.LeakyReLU(negative_slope=0.2),
        #         nn.Linear(latent_size // 4 * 2 * 2, latent_size // 4 * 4 * 4),
        #         nn.LeakyReLU(negative_slope=0.2),
        #     )

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            if i > 0:
                self.up_sample_layers += [ConvUpBlock(ch * 2, ch // 2)]
                ch //= 2
            else:
                self.up_sample_layers += [ConvUpBlock(ch * 2, ch)]

        self.up_sample_layers += [ConvUpBlock(ch * 2, ch)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        )

    def forward(self, z, cond):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = cond
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output, skip_out = layer(output)
            stack.append(skip_out)

        # z_out = self.middle_z_grow_linear(z)
        #
        # if self.resolution == 384:
        #     z_out = torch.reshape(z_out, (output.shape[0], self.latent_size // 4, 12, 12))
        # else:
        #     z_out = torch.reshape(z_out, (output.shape[0], self.latent_size // 4, 4, 4))

        # z_out = self.middle_z_grow_conv(z_out)
        # output = torch.cat([z_out, output], dim=1)
        output = self.res_layer_1(output)
        # output = torch.cat([z_out, output], dim=1)
        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = layer(output, stack.pop())

        return self.conv2(output)
