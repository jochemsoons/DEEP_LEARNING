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

# Function for setting the seed (copied from the notebook tutorials)
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

def train(config, seed):
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    set_seed(seed)
    print("Device:", device)
    print("Seed: ", seed)

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
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)

        # Compute the loss, gradients and update network parameters
        loss = loss_function(log_probs, batch_targets)
        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / log_probs.size(0)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

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

        # Check if training is finished
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            test_steps = config.test_size // config.batch_size
            print(test_steps)
            test_accuracy = test_model(model, test_steps, data_loader, device)
            print("TEST ACC:{:.3f}".format(test_accuracy))
            break

    print('Done training.')
    pickle.dump(np.asarray(loss_list), open('./plotdata/loss_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    pickle.dump(np.asarray(acc_list), open('./plotdata/acc_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    pickle.dump(np.asarray(steps_list), open('./plotdata/steps_{}_{}_{}.sav'.format(config.model_type, seq_length, seed), 'wb'))
    ###########################################################################
    ###########################################################################

def test_model(model, test_steps, data_loader, device):
    acc_list = []
    model.eval()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]

        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)

        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        acc = correct / log_probs.size(0)
        acc_list.append(acc)
        if step+1 >= test_steps:
            break
    return np.mean(acc_list)

def plot_results(model, eval_method, seq_lengths, seeds):
    plot_color = 'b'
    if eval_method == 'loss':
        title = 'loss'
        y_label = "Loss"
    elif eval_method == 'acc':
        title = 'accuracy'
        y_label = "Accuracy"
    for seq_length in seq_lengths:
        eval_lists = []
        steps_lists = []
        label = "T={} (k={})".format(seq_length, len(seeds))
        for seed in seeds:
            plt.figure("individual {}".format(eval_method))
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

        plt.figure("mean {}".format(eval_method))
        eval_lists = np.asarray(eval_lists)
        mean_eval_lists = np.mean(eval_lists, axis=0)
        std_dev = np.std(eval_lists, axis=0)
        plt.plot(steps_list, mean_eval_lists, c=plot_color, label="T={}".format(seq_length))
        plt.fill_between(steps_list, mean_eval_lists+std_dev, mean_eval_lists-std_dev, alpha=0.25, color=plot_color)
        plt.title("Averaged {} plots of {} model (bipalindrome dataset)".format(title, model))
        plt.xlabel("Train step")
        plt.ylabel(y_label)
        plt.legend(loc='best')
        plot_color = 'r'
    plt.figure("individual {}".format(eval_method))
    plt.savefig("./plots/{}_{}_separate".format(model, eval_method))
    plt.figure("mean {}".format(eval_method))
    plt.savefig("./plots/{}_{}_averaged".format(model, eval_method))
    # plt.show()




if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Seed
    parser.add_argument('--seeds', type=list, default=[0, 1, 2],
                        help='Seed for reproducibilty')
    # Plot results or not.
    parser.add_argument('--plot', type=bool, default=False,
                        help='Choose whether to plot results or not')

    # Train model or not
    parser.add_argument('--train', type=bool, default=False,
                        help='Choose whether to train or not')

    # dataset
    parser.add_argument('--dataset', type=str, default='bipalindrome',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
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

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--test_size', type=int, default=5000,
                        help='Number of testing samples')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    config = parser.parse_args()
    print(config.train)
    if config.train:
        for seq_length in [10, 20]:
            for seed in config.seeds:
                config.input_length = seq_length
                train(config, int(seed))
    if config.plot:
        plot_results('LSTM', 'loss', [10, 20], [0, 1, 2])
        plot_results('LSTM', 'acc', [10, 20], [0, 1, 2])

    # Train the model


