"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
from modules import LinearModule
import cifar10_utils

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
    preds = np.argmax(predictions, axis=1)
    labels = np.argmax(targets, axis=1)
    accuracy = np.mean(preds == labels)
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
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)

    depth, width, height = cifar10['train'].images[0].shape
    n_inputs = depth * width * height
    n_classes = len(cifar10['train'].labels[0])

    MLP_classifier = MLP(n_inputs, dnn_hidden_units, n_classes)
    CE_module = CrossEntropyModule()
    loss_train = []
    loss_test = []
    acc_list = []
    eval_steps = []
    for step in range(FLAGS.max_steps+1):
        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        x_train = x_train.reshape(x_train.shape[0], -1)

        predictions = MLP_classifier.forward(x_train)

        train_loss = CE_module.forward(predictions, y_train)
        loss_train.append(train_loss)

        loss_grad = CE_module.backward(predictions, y_train)
        MLP_classifier.backward(loss_grad)

        for layer in MLP_classifier.layers:
            if isinstance(layer, LinearModule):
                # print("UPDATING WEIGHTS")
                layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
                layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']

        if step % FLAGS.eval_freq == 0:
            acc = 0
            test_loss = 0
            batch_count = 0
            current_epochs = cifar10['test'].epochs_completed
            while True:
                x_test, y_test = cifar10['test'].next_batch(FLAGS.batch_size)
                if cifar10['test'].epochs_completed > current_epochs:
                    cifar10['test']._index_in_epoch = 0
                    break
                x_test = x_test.reshape(x_test.shape[0], -1)
                predictions = MLP_classifier.forward(x_test)
                test_loss += CE_module.forward(predictions, y_test)
                acc += accuracy(predictions, y_test)
                batch_count += 1
            acc = acc / batch_count
            test_loss = test_loss / batch_count
            print(cifar10['test'].epochs_completed)
            # print(acc)
            # print(test_loss)
            eval_steps.append(step)
            loss_test.append(test_loss)
            acc_list.append(acc)



    x = np.arange(0, FLAGS.max_steps, 1)
    # x1 = np.arange(0, FLAGS.max_steps, 1)
    plt.plot(x, loss_train, label='train loss')
    plt.plot(eval_steps, loss_test, label='test loss')
    # plt.plot(eval_steps, acc_list, label='accuracy')
    plt.legend()
    plt.show()


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
