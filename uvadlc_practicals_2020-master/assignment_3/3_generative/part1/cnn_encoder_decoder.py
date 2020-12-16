################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network

        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the encoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        act_fn = nn.GELU
        self.mean_encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2), # 28x28 => 16x16
            act_fn(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*num_filters, z_dim)
        )
        self.logstd_encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2), # 28x28 => 16x16
            act_fn(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*num_filters, z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        mean = self.mean_encoder(x)
        log_std = self.logstd_encoder(x)
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.

        Inputs:
            num_input_channels- Number of channels of the image to
                                reconstruct. For MNIST, this parameter is 3
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        act_fn = nn.ReLU
        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2*16*num_filters),
            act_fn()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*num_filters, 2*num_filters, kernel_size=3, output_padding=0, padding=1, stride=2), # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2*num_filters, 2*num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*num_filters, num_filters, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.linear(z)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1, 4, 4)
        # print(x.shape)
        x = self.decoder(x)
        # print(x.shape)
        # exit()
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device