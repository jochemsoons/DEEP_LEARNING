"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        embedding_dim = 2 * seq_length
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        # Input modulation gate.
        self.W_gx = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.zeros(hidden_dim))

        # Input gate.
        self.W_ix = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))

        # Forget gate.
        self.W_fx = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim))

        # Output gate
        self.W_ox = nn.Parameter(torch.Tensor(embedding_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))

        # Linear layer
        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_p = nn.Parameter(torch.zeros(num_classes))

        # Initialize weights with kaiming normal with their respective activation function.
        self.init_kaiming([self.W_gx, self.W_gh], 'tanh')
        self.init_kaiming([self.W_ix, self.W_ih, self.W_fx, self.W_fh, self.W_ox, self.W_oh], 'sigmoid')
        self.init_kaiming([self.W_ph], 'linear')
        # Initialize embedding and softmax layer.
        self.embedding = nn.Embedding(3, embedding_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    # Helper function to initialize a list of tensors with the nn.init.kaiming_normal function.
    def init_kaiming(self, tensors, activation):
        for weight in tensors:
            nn.init.kaiming_normal_(weight, nonlinearity=activation)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # Squeeze the tensor to remove the last dimension and embed the input.
        x = x.squeeze()
        x = self.embedding(x)

        # Initialize the hidden state and cell state.
        h_t = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)

        # Loop over the sequence and update the cell state and hidden state.
        for t in range(self.seq_length):
            x_t = x[:, t, :]
            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            i_t = torch.sigmoid(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            o_t = torch.sigmoid(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        # Produce the final output through the linear and softmax layer.
        p_t =  h_t @ self.W_ph + self.b_p
        return self.softmax(p_t)
        ########################
        # END OF YOUR CODE    #
        #######################

