"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_clases = n_classes
        self.layers = []
        layer_id = 0

        in_features = n_inputs
        for n_units in n_hidden:
          self.layers.append(LinearModule(in_features, n_units))
          self.layers.append(ELUModule())
          in_features = n_units

        output_layer = LinearModule(in_features, n_classes)
        self.layers.append(output_layer)
        self.layers.append(SoftMaxModule())

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
        input_ = x
        for layer in self.layers:
          # print(layer)
          # print(type(layer))
          # print("INPUT:", input_.shape)
          out = layer.forward(input_)
          # print("OUTPUT:", out.shape)
          # print(out)
          input_ = out
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in reversed(self.layers):
          # print(layer)
          # print("PREV:", dout.shape)
          dx = layer.backward(dout)
          # print("CURRENT:", dx.shape)
          dout = dx
        ########################
        # END OF YOUR CODE    #
        #######################

        return
