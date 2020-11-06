"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    preds = torch.argmax(predictions, axis=1)
    labels = torch.argmax(targets, axis=1)
    # print((preds == labels))
    # print(torch.sum(preds == labels).float())
    # print(preds.size()[0])
    accuracy = torch.sum(preds == labels).float() / preds.shape[0]

    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device, flush=True)

    # GPU operations have a separate seed we also want to set

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    depth, width, height = cifar10['train'].images[0].shape
    n_inputs = depth * width * height
    n_classes = len(cifar10['train'].labels[0])

    MLP_classifier = MLP(n_inputs, dnn_hidden_units, n_classes)
    # optimizer = torch.optim.SGD(MLP_classifier.parameters(), lr=FLAGS.learning_rate)
    optimizer = torch.optim.Adam(MLP_classifier.parameters(), lr=FLAGS.learning_rate)
    # loss_module = nn.NLLLoss()
    loss_module = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        MLP_classifier.to(device)
        loss_module.to(device)

    print(MLP_classifier, flush=True)
    loss_train = []
    avg_train_loss_list = []
    loss_test = []
    acc_list = []
    eval_steps = []
    num_epochs = 0
    for step in range(FLAGS.max_steps):
        MLP_classifier.train()

        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_train, y_train = x_train.to(device), y_train.to(device)

        x_train = x_train.reshape(x_train.size(0), -1)
        predictions = MLP_classifier.forward(x_train)
        # print("\nCALCULATING LOSS")
        labels = torch.argmax(y_train, dim=1)
        train_loss = loss_module(predictions, labels)

        loss_train.append(train_loss)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # if step % FLAGS.eval_freq == 0:
        if cifar10['train']._epochs_completed > num_epochs:
            MLP_classifier.eval()
            acc = 0
            test_loss = 0
            batch_count = 0
            current_epochs = cifar10['test'].epochs_completed
            while True:
                x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
                if cifar10['test'].epochs_completed > current_epochs:
                    cifar10['test']._index_in_epoch = 0
                    break
                x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
                x_test, y_test = x_test.to(device), y_test.to(device)

                x_test = x_test.reshape(x_test.size(0), -1)
                predictions = MLP_classifier.forward(x_test)
                labels = torch.argmax(y_test, dim=1)
                test_loss += loss_module(predictions, labels)
                acc += accuracy(softmax(predictions), y_test)
                batch_count += 1

            acc = acc / batch_count
            test_loss = test_loss / batch_count

            print("Train epoch: {} Test epoch: {}".format(cifar10['train'].epochs_completed, cifar10['test'].epochs_completed))

            # print(acc)
            # print(test_loss)
            eval_steps.append(step)
            avg_train_loss = torch.mean(train_loss)
            # print(avg_train_loss)
            avg_train_loss_list.append(avg_train_loss)
            print("test acc:{:.4f}, test loss:{:.4f}, train loss:{:.4f}".format(float(acc), float(test_loss), float(avg_train_loss)))
            train_loss = []
            loss_test.append(test_loss)
            acc_list.append(acc)
            num_epochs += 1
            # MLP_classifier.train()

    x = np.arange(0, num_epochs, 1)
    # x1 = np.arange(0, FLAGS.max_steps, 1)
    plt.figure()
    plt.plot(x, avg_train_loss_list, label='train loss')
    plt.plot(x, loss_test, label='test loss')
    # plt.plot(x, acc_list, label='accuracy')
    plt.legend()
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(x, acc_list, label='accuracy')
    plt.legend()
    plt.savefig("acc.png")

    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
