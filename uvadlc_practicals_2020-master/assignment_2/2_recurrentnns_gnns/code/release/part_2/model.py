# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        embedding_dim = 2 * seq_length
        self.device = device
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        # Initialize LSTM model.
        self.LSTM = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers)
        # Initialize final linear output layer.
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

    # NOTE: I added the states argument for efficiency purposes in text generation.
    # states = (hidden_state, cell_state)
    def forward(self, x, states=None):
        # Embed input.
        x = self.embedding(x)
        # Give input to LSTM model.
        h, states = self.LSTM(x, states)
        # Return the output and (hidden_state, cell_state).
        p = self.linear(h)
        return p, states
