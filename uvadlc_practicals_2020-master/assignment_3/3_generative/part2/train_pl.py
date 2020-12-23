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

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mnist import mnist
from models import GeneratorMLP, DiscriminatorMLP


class GAN(pl.LightningModule):

    def __init__(self, hidden_dims_gen, hidden_dims_disc, dp_rate_gen,
                 dp_rate_disc, z_dim, lr):
        """
        PyTorch Lightning module that summarizes all components to train a GAN.

        Inputs:
            hidden_dims_gen  - List of hidden dimensionalities to use in the
                              layers of the generator
            hidden_dims_disc - List of hidden dimensionalities to use in the
                               layers of the discriminator
            dp_rate_gen      - Dropout probability to use in the generator
            dp_rate_disc     - Dropout probability to use in the discriminator
            z_dim            - Dimensionality of latent space
            lr               - Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = GeneratorMLP(z_dim=z_dim,
                                         hidden_dims=hidden_dims_gen,
                                         dp_rate=dp_rate_gen)
        self.discriminator = DiscriminatorMLP(hidden_dims=hidden_dims_disc,
                                                 dp_rate=dp_rate_disc)

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images from the generator.

        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        sampled_z = torch.randn((batch_size, self.hparams.z_dim)).to(self.generator.device)
        x = self.generator(sampled_z)
        return x

    @torch.no_grad()
    def interpolate(self, batch_size, interpolation_steps):
        """
        Function for interpolating between a batch of pairs of randomly sampled
        images. The interpolation is performed on the latent input space of the
        generator.

        Inputs:
            batch_size          - Number of image pairs to generate
            interpolation_steps - Number of intermediate interpolation points
                                  that should be generated.
        Outputs:
            x - Generated images of shape [B,interpolation_steps+2,C,H,W]
        """
        # Create empty placeholder for x.
        x = torch.empty(batch_size, interpolation_steps + 2, 1, 28, 28)
        # Loop over the batch size of interpolations we need to make.
        for pair in range(batch_size):
            # Create pair of latent vectors.
            pair_z = torch.randn((2, self.hparams.z_dim)).to(self.generator.device)
            start = pair_z[0]
            end = pair_z[1]
            # Calculate interpolation step size.
            diff = (end - start) / (interpolation_steps + 1)
            # Create empty placeholder for the latent vectors at each interpolation step.
            latent_between = torch.empty(2 + interpolation_steps, self.hparams.z_dim)
            # First tensor is the starting latent vector.
            latent_between[0] = start
            current = start
            # Loop over the number of steps.
            for step in range(1, interpolation_steps+1):
                # Create latent vector at each interpolation step.
                latent_between[step] = current + diff
                current = current + diff
            # Final tensor is the ending latent vector.
            latent_between[step+1] = end
            # Compute the output images based on the latent vectors and assign to x.
            images = self.generator(latent_between.to(self.generator.device))
            x[pair] = images
        return x

    def configure_optimizers(self):
        # Create optimizer for both generator and discriminator.
        # You can use the Adam optimizer for both models.
        # It is recommended to reduce the momentum (beta1) to e.g. 0.5
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5,0.999))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5,0.999))
        return [optimizer_gen, optimizer_disc], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        The training step is called for every optimizer independently. As we
        have two optimizers (Generator and Discriminator), we therefore have
        two training steps for the same batch with different optimizer_idx.
        The loss returned for the optimizer_idx=0 should correspond to the loss
        of the first optimizer returned in configure_optimizers (i.e. by the
        generator). The second time the function is called (optimizer_idx=1),
        we optimize the discriminator. See the individual step functions
        "generator_step" and "discriminator_step" for their specific loss
        calculation.

        Inputs:
            batch         - Input batch from MNIST dataset
            batch_idx     - Index of the batch in the dataset (not needed here)
            optimizer_idx - Index of the optimizer to use for a specific
            training step - 0 = Generator, 1 = Discriminator
        """
        x, _ = batch

        if optimizer_idx == 0:
            loss = self.generator_step(x)
        elif optimizer_idx == 1:
            loss = self.discriminator_step(x)

        return loss

    def generator_step(self, x_real):
        """
        Training step for the generator. Note that you do *not* need to take
        any special care of the discriminator in terms of stopping the
        gradients to its parameters, as this is handled by having two different
        optimizers. Before the discriminator's gradients in its own step are
        calculated, the previous ones are set to zero by PyTorch Lightning's
        internal training loop. Remember to log the training loss.

        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the generator to optimize
        """
        batch_size = x_real.shape[0]
        # Sample but do not use the sample function because it contains no gradients.
        sampled_z = torch.randn((batch_size, self.hparams.z_dim)).to(self.generator.device)
        # Generate fake images.
        x_fake = self.generator(sampled_z)
        # Compute discriminator output.
        disc_fake = self.discriminator(x_fake)
        # Calculate generator loss using the BCE function.
        loss_gen = F.binary_cross_entropy_with_logits(torch.squeeze(disc_fake), torch.ones(batch_size).to(self.generator.device))
        self.log("generator/loss", loss_gen)
        return loss_gen

    def discriminator_step(self, x_real):
        """
        Training step for the discriminator. Note that you do not have to use
        the same generated images as in the generator_step. It is simpler to
        sample a new batch of "fake" images, and use those for training the
        discriminator. Remember to log the training loss, and other potentially
        interesting metrics.

        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the discriminator to optimize
        """

        # Remark: there are more metrics that you can add.
        # For instance, how about the accuracy of the discriminator?
        batch_size = x_real.shape[0]
        # Generate fake images.
        x_fake = self.sample(batch_size)
        # Compute discriminator output.
        disc_fake = self.discriminator(x_fake)
        disc_real = self.discriminator(x_real)
        # Calculate discriminator loss.
        loss_disc = (F.binary_cross_entropy_with_logits(torch.squeeze(disc_real), torch.ones(batch_size).to(self.generator.device))
                    + F.binary_cross_entropy_with_logits(torch.squeeze(disc_fake), torch.zeros(batch_size).to(self.generator.device)))
        # Compute softmax predictions and accuracy metric.
        preds_fake = torch.round(torch.sigmoid(disc_fake).squeeze())
        preds_real = torch.round(torch.sigmoid(disc_real).squeeze())
        acc_fake = (preds_fake == torch.zeros(batch_size).to(self.generator.device)).float().mean()
        acc_real = (preds_real == torch.ones(batch_size).to(self.generator.device)).float().mean()
        acc_disc = (acc_fake + acc_real) / 2
        self.log("discriminator/loss", loss_disc)
        self.log("discriminator/accuracy_real", acc_real)
        self.log("discriminator/accuracy_fake", acc_fake)
        self.log("discriminator/accuracy", acc_disc)
        return loss_disc


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=64, every_n_epochs=10, save_to_disk=False):
        """
        Callback for adding generations of your model to TensorBoard and/or
        save them to disk every N epochs across training.
        Inputs:
            batch_size     - Number of images to generate
            every_n_epochs - Only save images every N epochs (o.w. tensorboard
                             overhead gets quite large)
            save_to_disk   - If True, the sample and image means should be
                              saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the GAN.
        The generated samples images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer   - The PyTorch Lightning "Trainer" object.
            pl_module - The GAN model that is currently being trained.
            epoch     - The epoch number to use for TensorBoard logging and
                        saving of the files.
        """
        # Hints:
        # - You can access the logging directory path via trainer.logger.log_dir
        # - You can access the tensorboard logger via trainer.logger.experiment
        # - Use torchvision function "make_grid" to create a grid of multiple images
        # - Use torchvision function "save_image" to save an image grid to disk
        gen_samples = pl_module.sample(self.batch_size)
        gen_samples = make_grid(gen_samples)
        trainer.logger.experiment.add_image("Generated GAN images {}".format(epoch), gen_samples, epoch)
        if self.save_to_disk:
            save_image(gen_samples, "{}/generated_{}.png".format(trainer.logger.log_dir, epoch))


class InterpolationCallback(pl.Callback):

    def __init__(self, batch_size=4, interpolation_steps=5,
                 every_n_epochs=10, save_to_disk=False):
        """
        Callback for adding interpolations between two images to TensorBoard
        and/or save them to disk every N epochs across training.
        Inputs:
            batch_size          - Number of image pairs to interpolate between
            interpolation_steps - Number of interpolation steps to perform
                                  between the two random images.
            every_n_epochs      - Only save those images every N epochs
                                   (otherwise tensorboard gets quite large)
            save_to_disk        - If True, the samples and image means should
                                  be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.interpolation_steps = interpolation_steps
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save the interpolations from the GAN.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer     - The PyTorch Lightning "Trainer" object.
            pl_module   - The GAN model that is currently being trained.
            epoch       - The epoch number to use for TensorBoard logging
                          and saving of the files.
        """
        # Hints:
        # - You can access the logging directory path via trainer.logger.log_dir
        # - You can access the tensorboard logger via trainer.logger.experiment
        # - Use the torchvision function "make_grid" to create a grid of multiple images
        # - Use the torchvision function "save_image" to save an image grid to disk

        # You also have to implement this function in a later question of the assignemnt.
        # By default it is skipped to allow you to test your other code so far.
        imgs = pl_module.interpolate(self.batch_size, self.interpolation_steps)
        # Reshape the images so we create a grid.
        interpolated_images = imgs.view(imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4])
        # Create the image grid.
        img_grid = make_grid(interpolated_images, nrow=self.interpolation_steps+2)
        trainer.logger.experiment.add_image("Interpolated GAN images {}".format(epoch), img_grid, epoch)
        if self.save_to_disk:
            save_image(img_grid, "{}/interpolated_{}.png".format(trainer.logger.log_dir, epoch))


def train_gan(args):
    """
    Function for training and testing a GAN model.
    The function is ready for usage. Feel free to adjust it if wanted.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader = mnist(batch_size=args.batch_size,
                         num_workers=args.num_workers)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(save_to_disk=True)
    inter_callback = InterpolationCallback(save_to_disk=True)
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[gen_callback,
                                    inter_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    # Create model
    pl.seed_everything(args.seed)  # To be reproducable
    model = GAN(hidden_dims_gen=args.hidden_dims_gen,
                hidden_dims_disc=args.hidden_dims_disc,
                dp_rate_gen=args.dp_rate_gen,
                dp_rate_disc=args.dp_rate_disc,
                z_dim=args.z_dim,
                lr=args.lr)

    if not args.progress_bar:
        print("\nThe progress bar has been surpressed. For updates on the training progress, " + \
              "check the TensorBoard file at " + trainer.logger.log_dir + ". If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Training
    inter_callback.sample_and_save(trainer, model, epoch=0)
    gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader)

    return model


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims_gen', default=[128, 256, 512],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' + \
                             'generator. To specify multiple, use " " to ' + \
                             'separate them. Example: \"128 256 512\"')
    parser.add_argument('--hidden_dims_disc', default=[512, 256],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' + \
                             'discriminator. To specify multiple, use " " to ' + \
                             'separate them. Example: \"512 256\"')
    parser.add_argument('--dp_rate_gen', default=0.1, type=float,
                        help='Dropout rate in the discriminator')
    parser.add_argument('--dp_rate_disc', default=0.3, type=float,
                        help='Dropout rate in the discriminator')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size to use for training')

    # Other hyperparameters
    parser.add_argument('--epochs', default=250, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' + \
                             'To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='GAN_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' + \
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. '+ \
                             'Not to be used in conjuction with SLURM jobs.')

    args = parser.parse_args()

    train_gan(args)
