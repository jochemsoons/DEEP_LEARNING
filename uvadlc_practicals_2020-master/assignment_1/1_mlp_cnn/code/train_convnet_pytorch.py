"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    accuracy = torch.sum(preds == labels).float() / preds.shape[0]

    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    # GPU operations have a separate seed we also want to set

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    depth, width, height = cifar10['train'].images[0].shape
    n_inputs = depth * width * height
    n_classes = len(cifar10['train'].labels[0])

    CNN = ConvNet(3, n_classes)
    optimizer = torch.optim.Adam(CNN.parameters(), lr=FLAGS.learning_rate)
    loss_module = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    if torch.cuda.is_available():
        print("CUDA")
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        CNN.to(device)
        loss_module.to(device)

    train_loss_list, test_loss_list = [], []
    test_acc_list = []
    for step in range(FLAGS.max_steps):
        CNN.train()

        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        # x_train, y_train = cifar10['train'].next_batch(5)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        x_train, y_train = x_train.to(device), y_train.to(device)

        # x_train = x_train.reshape(x_train.size(0), -1)
        predictions = CNN.forward(x_train)
        # print("\nCALCULATING LOSS")
        labels = torch.argmax(y_train, dim=1)
        train_loss = loss_module(predictions, labels)
        # print(train_loss)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == 0:
            CNN.eval()
            acc = 0
            test_loss = 0
            batch_count = 0
            current_epochs = cifar10['test'].epochs_completed
            while True:
                with torch.no_grad():
                    x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
                    if cifar10['test'].epochs_completed > current_epochs:
                        cifar10['test']._index_in_epoch = 0
                        break
                    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
                    x_test, y_test = x_test.to(device), y_test.to(device)

                    predictions = CNN.forward(x_test)
                    labels = torch.argmax(y_test, dim=1)
                    test_loss += loss_module(predictions, labels)
                    acc += accuracy(softmax(predictions), y_test)
                    batch_count += 1

            acc = acc / batch_count
            test_loss = test_loss / batch_count

            print("Train epoch: {} Test epoch: {}".format(cifar10['train'].epochs_completed, cifar10['test'].epochs_completed))
            print("test acc:{:.4f}, test loss:{:.4f}, train loss:{:.4f}".format(float(acc), float(test_loss), float(train_loss)))

            test_loss_list.append(test_loss)
            test_acc_list.append(acc)

    ########################
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
