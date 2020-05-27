import numpy as np
import h5py
from convert_to_one_hot import convert_to_one_hot


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # Flatten the training and test images
    X_train_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    X_test_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    
    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(train_set_y_orig, classes.shape[0])
    Y_test = convert_to_one_hot(test_set_y_orig, classes.shape[0])
    
    return X_train, Y_train,X_test, Y_test, classes

