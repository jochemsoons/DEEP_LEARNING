"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
            self.PreActBlock(64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.PreActBlock(128),
            self.PreActBlock(128),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.PreActBlock(256),
            self.PreActBlock(256),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.PreActBlock(512),
            self.PreActBlock(512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.PreActBlock(512),
            self.PreActBlock(512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, n_classes),
        )

    class PreActBlock(nn.Module):
        """
        A inner class of the pre-activation ResNetBlock that I created.
        It can be initiliazed given the number of input channels,
        and in the forward function it returns the output of the block + the input.

        """
        def __init__(self, n_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            )
        def forward(self, x):
            out = x + self.block(x)
            return out

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = self.layers(x)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
