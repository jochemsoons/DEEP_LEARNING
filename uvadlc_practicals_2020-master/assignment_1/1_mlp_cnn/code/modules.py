"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.params = {}
        self.grads = {}
        self.params['weight'] = np.random.normal(0, 0.0001, size=(out_features, in_features))
        self.params['bias'] = np.zeros(out_features)
        self.params['grads'] = 0

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.input = x
        out = np.dot(self.input, self.params['weight'].T) + self.params['bias']
        self.output = out
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['weight'] = np.dot(dout.T, self.input)
        # dout.T.dot(self.input)
        # print(np.ones((self.params['bias'].shape)).shape)
        # print(dout.shape)
        # print(self.input.shape)
        # print(self.output.shape)
        self.grads['bias'] = np.dot(np.ones(self.input.shape[0]).T, dout)

        dx = np.dot(dout, self.params['weight'])

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # self.input = x
        # b = np.max(x, axis=0)
        # Y = np.exp(x - b)
        # out = Y / np.sum(Y, axis=1)
        # self.output = out

        self.input = x
        # z = x
        e = np.exp(x-np.max(x))
        # print()
        s = np.sum(e, axis=1, keepdims=True)
        # print("S", s)
        out = e / s
        self.output = out
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        S, C = self.input.shape
        softmax_prod = np.einsum('ij,ik->ijk', self.output, self.output)
        softmax_diag = np.einsum('ij,jk->ijk', self.output, np.eye(C, C))
        d_softmax = softmax_diag - softmax_prod
        # print(d_softmax.shape)
        # print(dout.shape)
        dx = np.einsum('ijk,ik->ij', d_softmax, dout)



        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        S, _ = x.shape
        # x = np.clip(x, epsilon, 1 - epsilon)
        out = - np.sum(y * np.log(x + 1e-9)) / S
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        S, _ = x.shape
        dx = - y / x * 1/S

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


# class ELUModule(object):
#     """
#     ELU activation module.
#     """

#     def forward(self, x):
#         """
#         Forward pass.

#         Args:
#           x: input to the module
#         Returns:
#           out: output of the module

#         TODO:
#         Implement forward pass of the module.

#         Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
#         """
#         ########################
#         # PUT YOUR CODE HERE  #
#         #######################
#         self.input = x
#         out = np.maximum(0,x)
#         return out

#         ########################
#         # END OF YOUR CODE    #
#         #######################

#         return out

#     def backward(self, dout):
#         """
#         Backward pass.
#         Args:
#           dout: gradients of the previous module
#         Returns:
#           dx: gradients with respect to the input of the module

#         TODO:
#         Implement backward pass of the module.
#         """

#         ########################
#         # PUT YOUR CODE HERE  #
#         #######################

#         relu_grad = self.input > 0
#         return dout * relu_grad

#         ########################
#         # END OF YOUR CODE    #
#         #######################

class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.input = x
        out = np.where(x > 0, x, np.exp(x)-1)

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        dh_dx = np.where(self.input > 0, 1, np.exp(self.input))
        dx = dout * dh_dx

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx