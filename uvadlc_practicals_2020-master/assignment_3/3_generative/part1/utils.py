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
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    eps = torch.randn_like(std)
    z = mean + eps*std
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    std = log_std.exp()
    KLD = 0.5 * torch.sum(std**2 + mean**2 - 1 - torch.log(std**2), dim=-1)
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    bpd = elbo * np.log2(np.exp(1)) / np.prod(img_shape[1:])
    bpd = torch.mean(bpd)
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/(grid_size+1)
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use scipy's function "norm.ppf" to obtain z values at percentiles.
    # - Use the range [0.5/(grid_size+1), 1.5/(grid_size+1), ..., (grid_size+0.5)/(grid_size+1)] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a sigmoid after the decoder
    x_range = torch.linspace(0.5/(grid_size+1), (grid_size+0.5)/(grid_size+1), steps=grid_size)
    y_range = torch.linspace(0.5/(grid_size+1), (grid_size+0.5)/(grid_size+1), steps=grid_size)
    z_x = torch.from_numpy(norm.ppf(x_range))
    z_y = torch.from_numpy(norm.ppf(y_range))
    # Create the grid based on the percentile values.
    grid_x, grid_y = torch.meshgrid(z_x, z_y)
    # Transform the list in a coordinate pair-wise list.
    z_list = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()
    # Compute the sampled images and create the manifold grid.
    x_mean = torch.sigmoid(decoder(z_list.to(decoder.device)))
    img_grid = make_grid(x_mean, nrow=grid_size)
    return img_grid
