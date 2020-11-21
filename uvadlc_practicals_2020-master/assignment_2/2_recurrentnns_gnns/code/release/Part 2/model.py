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
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.LSTM = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.softmax = nn.Softmax(dim=2)
        self.device = device

    def forward(self, x):
        # Implementation here...
        # x = x.to(torch.double)
        x = self.embedding(x)
        # print(x.shape)
        h, _ = self.LSTM(x)
        p = self.linear(h)
        return p
