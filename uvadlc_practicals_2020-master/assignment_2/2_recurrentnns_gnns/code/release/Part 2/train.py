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
import pickle
import matplotlib.pyplot as plt

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################
# Function for setting the seed (copied/adapted from the notebook tutorials)
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

def eval_acc(batch_output, batch_targets, criterion, config):
    acc = 0
    for t in range(config.seq_length):
        predictions = torch.argmax(batch_output[t], dim=1)
        correct = (predictions == batch_targets[t]).sum().item()
        acc += correct / config.batch_size
    accuracy = acc / config.seq_length
    return accuracy

def generate_sentence(model, sentence_length, vocab_length, temp=None):
    start = random.randint(0, vocab_length-1)
    state = None
    sentence = [start]
    for _ in range(sentence_length-1):
        input_tensor = torch.LongTensor([sentence[-1]]).unsqueeze(-1).to(model.device)
        out, state = model(input_tensor, hid_state=state)
        if temp:
            probs = F.softmax(temp*out, dim=2)
            char = torch.multinomial(probs[-1], num_samples=1, replacement=True)
        else:
            probs = F.softmax(out, dim=2)
            char = torch.argmax(probs[-1])
        sentence.append(int(char))
    return sentence

def finish_sentence(model, sentence, sentence_length, dataset, temp=None):
    input_length = len(sentence)
    assert sentence_length > input_length
    for i in range(sentence_length - input_length):
        if i == 0:
            input_tensor = torch.LongTensor(sentence).unsqueeze(-1).to(model.device)
            state = None
        else:
            input_tensor = torch.LongTensor([sentence[-1]]).unsqueeze(-1).to(model.device)
        out, state = model(input_tensor, hid_state=state)
        if temp:
            probs = F.softmax(temp*out, dim=2)
            char = torch.multinomial(probs[-1], 1)
        else:
            probs = F.softmax(out, dim=2)
            char = torch.argmax(probs[-1])
        sentence.append(int(char))
    return sentence

def write_sentences(out_file, num_sentences, step, config, model, dataset, tau=None):
    f = open(out_file, "a")
    # Generate some sentences by sampling from the model
    f.write("\nGenerated sentences at step {:04d}/{:04d}:\n".format(step, config.train_steps))
    f.write("TEMP = {}".format(tau))
    for length in [15, 30, 45, 60, 120]:
        f.write("\n### LENGTH {} ###\n".format(length))
        for i in range(num_sentences):
            sentence = generate_sentence(model, length, dataset.vocab_size, temp=tau)
            f.write("{}) {}\n".format(i+1, dataset.convert_to_string(sentence)))
    f.write("{}\n".format("#" * 120))
    return

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)
    # set_seed(42)
    print("Device:", device)
    # print("Seed:", seed)
    print("Txt file:", config.txt_file)
    f = open("sentences.txt", "w")
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(batch_size=config.batch_size, seq_length=config.seq_length,
            vocabulary_size=dataset.vocab_size, device=device).to(device)  # FIXME

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)  # FIXME

    loss_list, acc_list, steps_list = [], [], []
    step = 0
    for epoch in range(config.max_epochs):
        steps = step
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            step += steps
            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################

            batch_inputs = torch.stack(batch_inputs).to(device)
            batch_targets = torch.stack(batch_targets).to(device)

            output, _ = model(batch_inputs)

            # Reset for next iteration
            model.zero_grad()
            loss = criterion(output.permute(1, 2, 0), batch_targets.permute(1, 0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if (step + 1) % config.print_every == 0:
                accuracy = eval_acc(output, batch_targets, criterion, config)
                print("[{}] EPOCH {} Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                        ))
                loss_list.append(float(loss))
                acc_list.append(accuracy)
                steps_list.append(step)

            if (step + 1) % config.sample_every == 0:
                model.eval()
                write_sentences("sentences.txt", 10, step, config, model, dataset, tau=None)
                model.train()

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print('Done training.')
                torch.save(model.state_dict(), "./saved_models/model_{}.pt".format(config.train_steps))
                pickle.dump(np.asarray(loss_list), open('./plotdata/loss_{}.sav'.format(config.train_steps), 'wb'))
                pickle.dump(np.asarray(acc_list), open('./plotdata/acc_{}.sav'.format(config.train_steps), 'wb'))
                pickle.dump(np.asarray(steps_list), open('./plotdata/steps_{}.sav'.format(config.train_steps), 'wb'))
                return


def plot(plotdata_dir, eval_method, train_steps):
    plt.figure()
    if eval_method == 'loss':
        title = "loss"
        y_label = "Cross-Entropy loss"
    elif eval_method == 'acc':
        title = "accuracy"
        y_label = "Accuracy"
    eval_list = pickle.load(open('{}/{}_{}.sav'.format(plotdata_dir, eval_method, train_steps), 'rb'))
    steps_list = pickle.load(open('{}/steps_{}.sav'.format(plotdata_dir, train_steps), 'rb'))
    plt.plot(steps_list, eval_list, c='b')
    plt.title("Train {} (Darwins reis om de wereld)".format(title))
    plt.xlabel("Train step")
    plt.ylabel(y_label)
    plt.savefig("plots/{}_{}".format(eval_method, train_steps))

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
    parser.add_argument('--lstm_num_hidden', type=int, default=512,
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

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=50,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=2000,
                        help='How often to sample from the model')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Choose whether to plot or not (note: adapt lines below according to files in plotdata)')
    parser.add_argument('--train', type=bool, default=False,
                        help='Choose whether to train from start or not')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Load model state dict.')
    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()
    print(config)
    if config.train:
        train(config)
    if config.plot:
        plot("./plotdata", "acc", 40000)
        plot("./plotdata", "loss", 40000)
    if config.load_model:
        # Initialize the dataset and data loader (note the +1)
        dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
        data_loader = DataLoader(dataset, config.batch_size)

        # Initialize the model that we are going to use
        model = TextGenerationModel(batch_size=config.batch_size, seq_length=config.seq_length,
                vocabulary_size=dataset.vocab_size, device=config.device).to(config.device)

        model.load_state_dict(torch.load(config.load_model, map_location=torch.device('cpu')))
        model.eval()
        f = open("temp_sentences.txt", "w")
        write_sentences("temp_sentences.txt", 10, 0, config, model, dataset, tau=None)
        write_sentences("temp_sentences.txt", 10, 0, config, model, dataset, tau=0.5)
        write_sentences("temp_sentences.txt", 10, 0, config, model, dataset, tau=1)
        write_sentences("temp_sentences.txt", 10, 0, config, model, dataset, tau=2)
        f = open("finished_sentences.txt", "w")
        unfinished_sents = ['Sleeping beauty is', 'Somewhere far away', 'Once upon a time',
                            'Democracy is', 'President Trump is', 'I vote for',
                            'Evolutie is', 'Menschen zijn', 'Menschen stammen af van']
        for unfinished_sent in unfinished_sents:
            unfinished_input = dataset.convert_to_ints(unfinished_sent)
            sentence = finish_sentence(model, unfinished_input, 80, dataset, temp=2)
            f.write("{}\n".format(dataset.convert_to_string(sentence)))

