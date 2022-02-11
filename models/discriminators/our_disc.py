"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NOTE: z_location tells the network where to use the latent variable. It has options:
    0: No latent vector
    1: Add latent vector to zero filled areas
    2: Add latent vector to middle of network (between encoder and decoder)
    3: Add as an extra input channel
"""

import torch
from torch import nn


class FullDownBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.in_chans, self.out_chans, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """

        return self.downsample(input)  # self.resblock(self.downsample(input))

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_chans}, out_chans={self.out_chans}\nResBlock(in_chans={self.out_chans}, out_chans={self.out_chans}'


class DiscriminatorModel(nn.Module):
    def __init__(self, in_chans, out_chans, resolution):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        # CHANGE BACK TO 16 FOR MORE
        self.encoder_layers = nn.ModuleList()
        if resolution == 384:
            self.initial_layers = nn.Sequential(
                nn.Conv2d(self.in_chans*2, 16, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU()
            )
            self.encoder_layers += [FullDownBlock(16, 32)]
        else:
            self.initial_layers = nn.Sequential(
                nn.Conv2d(self.in_chans * 2, 32, kernel_size=(3, 3), padding=1),
                nn.LeakyReLU()
            )

        self.encoder_layers += [FullDownBlock(32, 64)]
        self.encoder_layers += [FullDownBlock(64, 128)]
        self.encoder_layers += [FullDownBlock(128, 256)]
        self.encoder_layers += [FullDownBlock(256, 512)]
        self.encoder_layers += [FullDownBlock(512, 512)]
        self.encoder_layers += [FullDownBlock(512, 512)]

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*3*3,1) if resolution == 384 else nn.Linear(512*2*2,1)
        )

    def forward(self, input, y):
        output = torch.cat([input,y], dim=1)
        output = self.initial_layers(output)

        # Apply down-sampling layers
        for layer in self.encoder_layers:
            output = layer(output)

        return self.dense(output)
