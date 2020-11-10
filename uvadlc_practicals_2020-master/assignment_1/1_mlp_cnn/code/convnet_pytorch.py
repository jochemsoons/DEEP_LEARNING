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

        self.conv0 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
        self.preact1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.preact2_a = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.preact2_b = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.preact3_a = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.preact3_b = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.preact4_a = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.preact4_b = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.preact5_a = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.preact5_b = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(512)
        self.ReLu = nn.ReLU()
        self.linear = nn.Linear(512, n_classes)
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
        out = self.conv0(x)
        out = out + self.preact1(out)
        out = self.conv1(out)
        out = self.maxpool1(out)

        out = out + self.preact2_a(out)
        out = out + self.preact2_b(out)

        out = self.conv2(out)
        out = self.maxpool1(out)

        out = out + self.preact3_a(out)
        out = out + self.preact3_b(out)

        out = self.conv3(out)
        out = self.maxpool1(out)

        out = out + self.preact4_a(out)
        out = out + self.preact4_b(out)

        out = self.maxpool1(out)

        out = out + self.preact5_a(out)
        out = out + self.preact5_b(out)
        out = self.maxpool1(out)

        out = self.batchnorm(out)
        out = self.ReLu(out)
        out = out.contiguous().view(out.size(0), -1)
        out = self.linear(out)

        ########################
        # END OF YOUR CODE    #
        #######################

        return out
