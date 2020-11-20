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
        input_dim = 256
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.W_gx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ix = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_fx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ox = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, num_classes))
        self.b_p = nn.Parameter(torch.Tensor(num_classes))
        self.embedding = nn.Embedding(3, input_dim)
        self.softmax = nn.LogSoftmax(dim=1)
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
        # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        x = self.embedding(x)
        # print(x.shape)

        h_t, c_t = (
                torch.zeros(self.batch_size, self.hidden_dim).to(x.device),
                torch.zeros(self.batch_size, self.hidden_dim).to(x.device),
            )
        # print(h_t.shape)

        for t in range(self.seq_length):
            # print(t)
            x_t = x[:, t, :]
            # print(x_t.shape)
            # printself.W_gx.shape)
            # print(h_t.shape)
            # print(self.W_gh.shape)
            g_t = torch.tanh(x_t @ self.W_gx + h_t @ self.W_gh + self.b_g)
            i_t = torch.tanh(x_t @ self.W_ix + h_t @ self.W_ih + self.b_i)
            f_t = torch.tanh(x_t @ self.W_fx + h_t @ self.W_fh + self.b_f)
            o_t = torch.tanh(x_t @ self.W_ox + h_t @ self.W_oh + self.b_o)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t
        p_t =  h_t @ self.W_ph + self.b_p
        y_t = self.softmax(p_t)

        ########################
        # END OF YOUR CODE    #
        #######################
        return y_t
