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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

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
    print("Device:", device)

    # Ensure all CUDA operations are deterministic.
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset.
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    # Determine number of input features and classes.
    depth, width, height = cifar10['train'].images[0].shape
    n_inputs = depth * width * height
    n_classes = len(cifar10['train'].labels[0])

    # Initialize MLP model, optimizer and CE loss module.
    MLP_classifier = MLP(n_inputs, dnn_hidden_units, n_classes)
    optimizer = torch.optim.Adam(MLP_classifier.parameters(), lr=FLAGS.learning_rate)
    loss_module = nn.CrossEntropyLoss()

    # Push model and loss module to device.
    MLP_classifier.to(device)
    loss_module.to(device)

    # Initialize evaluation lists.
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    train_loss_temp_list, train_acc_temp_list = [], []
    eval_steps = []

    # Iterate over each step.
    for step in range(1, FLAGS.max_steps+1):
        MLP_classifier.train()
        # Get new training batch and flatten image dimensions.
        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_train = x_train.reshape(x_train.size(0), -1)

        # Perform forward pass of MLP model.
        predictions = MLP_classifier.forward(x_train)

        # Calculate train loss and accuracy.
        labels = torch.argmax(y_train, dim=1)
        train_loss = loss_module(predictions, labels)
        train_acc = accuracy(predictions, y_train)
        # Add loss and acc. to the temporary lists.
        train_loss_temp_list.append(float(train_loss))
        train_acc_temp_list.append(float(train_acc))

        # Perform backward pass and update parameters.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Evaluate at eval frequency.
        if step % FLAGS.eval_freq == 0:
            MLP_classifier.eval()
            test_loss = test_acc = batch_count = 0
            current_epochs = cifar10['test'].epochs_completed
            # Keep getting new batches from test set.
            while True:
                x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
                # Stop evaluating if whole testset is passed.
                if cifar10['test'].epochs_completed > current_epochs:
                    cifar10['test']._index_in_epoch = 0
                    break
                with torch.no_grad():
                    # Add loss and accuracy to total using model predictions.
                    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    x_test = x_test.reshape(x_test.size(0), -1)
                    predictions = MLP_classifier.forward(x_test)
                    labels = torch.argmax(y_test, dim=1)
                    test_loss += loss_module(predictions, labels)
                    test_acc += accuracy(predictions, y_test)
                    batch_count += 1

            # Calculate average loss and acc. for train and test set.
            test_loss = test_loss / batch_count
            test_acc = test_acc / batch_count
            train_loss = np.mean(train_loss_temp_list)
            train_acc = np.mean(train_acc_temp_list)

            # Append evaluations to lists.
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            eval_steps.append(step)

            print("STEP {}/{} | test acc: {:.4f}, test loss: {:.4f} | train acc: {:.4f}, train loss: {:.4f}"
            .format(step, FLAGS.max_steps, test_acc, test_loss, train_acc, train_loss))

            # Reset temporary lists to calculate average train evaluations between eval freqs.
            train_acc_temp_list, train_loss_temp_list = [], []

    # Plot loss figure.
    plt.figure()
    plt.title("Train and test loss of PyTorch MLP model")
    plt.xlabel("Iteration step")
    plt.ylabel("Cross-entropy loss")
    plt.plot(eval_steps, train_loss_list, label='Train loss')
    plt.plot(eval_steps, test_loss_list, label='Test loss')
    plt.legend()
    plt.savefig("./MLP_pytorch_results/MLP_pytorch_loss.png")

    # Plot accuracy figure.
    plt.figure()
    plt.title("Train and test accuracy of PyTorch MLP model")
    plt.xlabel("Iteration step")
    plt.ylabel("Accuracy")
    plt.plot(eval_steps, train_acc_list, label="Train acc")
    plt.plot(eval_steps, test_acc_list, label="Test acc")
    plt.legend()
    plt.savefig("./MLP_pytorch_results/MLP_pytorch_acc.png")

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
