"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn
import torchvision.models as models

# Default constants
LEARNING_RATE_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'Adam'
MODEL_DEFAULT = 'DenseNet'

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

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

    # Determine number of input channels and classes.
    # n_channels, _, _ = cifar10['train'].images[0].shape
    n_classes = len(cifar10['train'].labels[0])
    print("MODEL:", MODEL_DEFAULT)
    # Initialize pre-trained model, optimizer and CE loss module.
    if MODEL_DEFAULT == 'DenseNet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, n_classes)
    elif MODEL_DEFAULT == 'ResNet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, n_classes)
    # for param in model.parameters():
    #         param.requires_grad = False
    # model.fc = nn.Linear(512, n_classes)
    # params_to_update = []
    # for param in model.parameters():
    #     if param.requires_grad == True:
    #         params_to_update.append(param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-4)
    loss_module = nn.CrossEntropyLoss()

    # Push model and loss module to device.
    model.to(device)
    loss_module.to(device)

    # Initialize evaluation lists.
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    train_loss_temp_list, train_acc_temp_list = [], []
    eval_steps = []
    num_epochs = 0
    # Iterate over each step.
    while num_epochs <= FLAGS.max_epochs:
        model.train()
        # Get new training batch and push to device.
        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
        x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
        x_train, y_train = x_train.to(device), y_train.to(device)

        # Perform forward pass of CNN model.
        predictions = model(x_train)

        # Calculate train loss and accuracy.
        labels = torch.argmax(y_train, dim=1)
        train_loss = loss_module(predictions, labels)
        # print(train_loss)
        train_acc = accuracy(predictions, y_train)
        # Add loss and acc. to the temporary lists.
        train_loss_temp_list.append(float(train_loss))
        train_acc_temp_list.append(float(train_acc))

        # Perform backward pass and update parameters.
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Evaluate at eval frequency.
        if cifar10['train'].epochs_completed > num_epochs:
            model.eval()
            test_loss, test_acc, batch_count = 0, 0, 0
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
                    predictions = model.forward(x_test)
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
            num_epochs += 1
            eval_steps.append(num_epochs)

            print("EPOCH {}/{} | test acc: {:.4f}, test loss: {:.4f} | train acc: {:.4f}, train loss: {:.4f}"
            .format(num_epochs, FLAGS.max_epochs, test_acc, test_loss, train_acc, train_loss))

            # Reset temporary lists to calculate average train evaluations between eval freqs.
            train_acc_temp_list, train_loss_temp_list = [], []


    # Plot loss figure.
    plt.figure()
    plt.title("Train and test loss of pretrained CNN model")
    plt.xlabel("Training epoch")
    plt.ylabel("Cross-entropy loss")
    plt.plot(eval_steps, train_loss_list, label='Train loss')
    plt.plot(eval_steps, test_loss_list, label='Test loss')
    plt.legend()
    plt.savefig("./CNN_pytorch_results/CNN_pretrained_loss.png")

    # Plot accuracy figure.
    plt.figure()
    plt.title("Train and test accuracy of pretrained CNN model")
    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy")
    plt.plot(eval_steps, train_acc_list, label="Train acc")
    plt.plot(eval_steps, test_acc_list, label="Test acc")
    plt.legend()
    plt.savefig("./CNN_pytorch_results/CNN_pretrained_acc.png")
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
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
