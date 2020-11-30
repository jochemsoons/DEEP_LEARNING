###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM

import numpy as np

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

###############################################################################

"""
Function for setting the seed (copied/adapted from the notebook tutorials).
"""
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

"""
Main training function.
"""
def train(config):
    # Initialize the device which to run the model on and set the seed.
    device = torch.device(config.device)
    set_seed(config.seed)
    print("Device:", device)
    print("Seed: ", config.seed)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        seq_length = config.input_length
        config.num_classes = 2
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        input_length = config.input_length*4+2-1


    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    loss_list, acc_list, steps_list = [], [], []
    convergence_count = 0
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network.
        t1 = time.time()

        # Move to GPU.
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration.
        model.zero_grad()

        # Forward pass.
        log_probs = model(batch_inputs)

        # Compute the loss, gradients and update network parameters.
        loss = loss_function(log_probs, batch_targets)
        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        # Calculate accuracy score.
        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / log_probs.size(0)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Eval step.
        if step % 60 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))
            loss_list.append(float(loss))
            acc_list.append(accuracy)
            steps_list.append(step)

            # Check for convergence criteria.
            if loss <= (0 + 0.005):
                convergence_count += 1
            # If no convergence reset convergence count.
            else:
                convergence_count = 0

        # Check if training is finished or if we have converged.
        if step == config.train_steps or convergence_count >= 3:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            test_steps = config.test_size // config.batch_size
            test_accuracy = test_model(model, test_steps, data_loader, device)
            break

    # Print final test accuracy and save evaluation lists (for printing).
    print('Done training.')
    print("TEST ACC:{:.3f}".format(test_accuracy))
    pickle.dump(np.asarray(loss_list), open('./plotdata/loss_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    pickle.dump(np.asarray(acc_list), open('./plotdata/acc_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    pickle.dump(np.asarray(steps_list), open('./plotdata/steps_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    ###########################################################################
    ###########################################################################

"""
Helper function to test the model for a given number of test steps (batches).
"""
def test_model(model, test_steps, data_loader, device):
    acc_list = []
    model.eval()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        log_probs = model(batch_inputs)
        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        acc = correct / log_probs.size(0)
        acc_list.append(acc)
        if step+1 >= test_steps:
            break
    return np.mean(acc_list)

"""
Helper function that pads evaluation lists so that they have the same length.
I implemented this function because I plot averaged accuracy and loss curves
for training runs of different length (due to early stopping).
Padding is perfomed by appending the final value of the lists x times so it
becomes the same length as the longest of the given lists.
"""
def pad_lists(eval_lists, steps_lists):
    list_lengths = [len(list_) for list_ in eval_lists]
    index = np.argmax(np.array(list_lengths))
    max_length = max(list_lengths)
    max_steps_list = steps_lists[index]
    padded_lists = []
    for i, eval_list in enumerate(eval_lists):
        if i != index:
            last_value = eval_list[-1]
            pad_length = max_length - len(eval_list)
            padded_list = np.pad(eval_list, (0, pad_length), 'constant', constant_values=(0, last_value))
            padded_lists.append(padded_list)
        else:
            padded_lists.append(eval_list)
    return np.asarray(padded_lists), max_steps_list

"""
Helper function to plot both the seperate training curves and the averaged training curves of
multiple training runs.
"""
def plot_results(model, eval_method, seq_lengths, seeds):
    plot_color = 'b'
    if eval_method == 'loss':
        title = 'loss'
        y_label = "Cross-Entropy loss"
    elif eval_method == 'acc':
        title = 'accuracy'
        y_label = "Accuracy"
    for seq_length in seq_lengths:
        eval_lists = []
        steps_lists = []
        label = "T={} (seeds={})".format(seq_length, len(seeds))
        for seed in seeds:
            # Load and plot the seperate seed-dependent eval lists.
            plt.figure("separate")
            eval_list = pickle.load(open('./plotdata/{}_{}_{}_{}.sav'.format(eval_method, model, seq_length, seed), 'rb'))
            steps_list = pickle.load(open('./plotdata/steps_{}_{}_{}.sav'.format(model, seq_length, seed), 'rb'))
            plt.plot(steps_list, eval_list, c=plot_color, label=label)
            label = "_nolegend_"
            plt.title("Separate {} plots of {} model (bipalindrome dataset)".format(title, model))
            plt.xlabel("Train step")
            plt.ylabel(y_label)

            plt.legend(loc='best')
            steps_lists.append(steps_list)
            eval_lists.append(eval_list)

        # Plot the averaged evaluation list with standard deviation.
        plt.figure("mean")
        # Pad the evaluation lists so they all have the same length.
        eval_lists, steps_list = pad_lists(eval_lists, steps_lists)
        mean_eval_lists = np.mean(eval_lists, axis=0)
        std_dev = np.std(eval_lists, axis=0)
        plt.plot(steps_list, mean_eval_lists, c=plot_color, label="T={}".format(seq_length))
        plt.fill_between(steps_list, mean_eval_lists+std_dev, mean_eval_lists-std_dev, alpha=0.25, color=plot_color)
        plt.title("Averaged {} plots of {} model (bipalindrome dataset)".format(title, model))
        plt.xlabel("Train step")
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plot_color = 'r'
    # Save the figures in the plots folder.
    plt.figure("separate")
    plt.savefig("./plots/{}_{}_separate".format(model, eval_method))
    plt.figure("mean")
    plt.savefig("./plots/{}_{}_averaged".format(model, eval_method))
    plt.close("all")


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Seed for reproducibility.
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for reproducibilty')
    # Plot results or not.
    parser.add_argument('--plot', action='store_true',
                        help='Specify whether to plot results or not')

    # Train (multiple) models or not
    parser.add_argument('--train', action='store_true',
                        help='Specify whether to train or not')
    parser.add_argument('--train_multiple', action='store_true',
                        help='Specify whether to train multiple models sequentially')

    # Dataset.
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params.
    parser.add_argument('--model_type', type=str, default='biLSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=10,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')

    # Training params.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--test_size', type=int, default=5000,
                        help='Number of testing samples')

    # Misc params.
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda:0"),
                        help="Device to run the model on.")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    config = parser.parse_args()

    # Perform single trainig run according to config.
    if config.train:
        train(config)

    # Perform multiple runs for the models, seq_lengths and seeds specified below.
    if config.train_multiple:
        for model in ['LSTM', 'peepLSTM']:
            config.model_type = model
            for seq_length in [10, 20]:
                config.input_length = seq_length
                for seed in [0, 1, 2]:
                    config.seed = seed
                    train(config)
    # Plot training curves (according to plotdata).
    if config.plot:
        plot_results('LSTM', 'loss', [10, 20], [0, 1, 2])
        plot_results('LSTM', 'acc', [10, 20], [0, 1, 2])
        plot_results('peepLSTM', 'loss', [10, 20], [0, 1, 2])
        plot_results('peepLSTM', 'acc', [10, 20], [0, 1, 2])


