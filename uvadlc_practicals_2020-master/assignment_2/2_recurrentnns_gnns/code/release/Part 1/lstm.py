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
        print("seq length:", seq_length)
        print("input dim:", input_dim)
        print("hidden dim:", hidden_dim)
        print("num classes:", num_classes)
        print("batch_size:", batch_size)
        print("device:", device)
        print("\n")

        self.W_gx = nn.Parameter(torch.Tensor(input_dim, num_classes))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_g = nn.Parameter(torch.Tensor(num_classes))

        self.W_ix = nn.Parameter(torch.Tensor(input_dim, num_classes))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_i = nn.Parameter(torch.Tensor(num_classes))

        self.W_fx = nn.Parameter(torch.Tensor(input_dim, num_classes))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_f = nn.Parameter(torch.Tensor(num_classes))

        self.W_ox = nn.Parameter(torch.Tensor(input_dim, num_classes))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_o = nn.Parameter(torch.Tensor(num_classes))
        self.embedding = nn.Embedding(seq_length, hidden_dim)
        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            try:
                nn.init.kaiming_normal_(weight)
            except:
                nn.init.zeros_(weight)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # print("FORWARD")
        # print(x.shape)
        # x = self.embedding(x)
        # print(x.shape)
        print(x[0])
        ########################
        # END OF YOUR CODE    #
        #######################
        return out
