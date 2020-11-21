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

import os
import time
from datetime import datetime
import argparse

import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################

def generate_sentence(model, length, vocab_length):
    model.eval()
    start = random.randint(0, vocab_length-1)
    state = None
    sentence = [start]
    for _ in range(length-1):
        input_tensor = torch.tensor([sentence[-1]]).unsqueeze(-1).to(model.device)
        out, state = model(input_tensor, state=state)
        char = torch.argmax(out[-1])
        sentence.append(int(char))
    model.train()
    return sentence
def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)
    print(config.txt_file)
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size, seq_length=config.seq_length,
            vocabulary_size=dataset.vocab_size, device=config.device).to(device)  # FIXME

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # FIXME

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################


        batch_inputs = torch.stack(batch_inputs).to(device)
        batch_targets = torch.stack(batch_targets).to(device)

        # batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        # batch_targets = batch_targets.to(device)   # [batch_size]
        # print(batch_targets.shape)
        output, _ = model(batch_inputs)
        # print(output.shape)
        # out_perm = output.permute(1,0,2)
        # print(out_perm.shape)
        # exit()

        # Reset for next iteration
        model.zero_grad()

        loss = acc = 0
        for t in range(config.seq_length):
            loss += criterion(output[t], batch_targets[t])   # fixme
            predictions = torch.argmax(output[t], dim=1)
            correct = (predictions == batch_targets[t]).sum().item()
            acc += correct / output[t].size(0)
        loss = loss / config.seq_length
        accuracy = acc / config.seq_length
        # print(loss, accuracy)
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            sentence = generate_sentence(model, 30, dataset.vocab_size)
            print(dataset.convert_to_string(sentence))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()
    print(config)
    # Train the model
    train(config)
