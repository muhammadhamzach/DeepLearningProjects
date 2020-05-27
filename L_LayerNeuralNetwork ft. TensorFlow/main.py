from tensor_model import tensor_model
from load_dataset import load_dataset
import numpy as np

X_train, Y_train, X_test, Y_test, classes = load_dataset()
m = X_train.shape[1]

layer_dims = [X_train.shape[0], 25, 12, np.shape(classes)[0]]  #dimesions of the neural network layer
learning_rate = 0.0001                      #alpha learning rate of the system
num_epochs = 1500                           #number of iterations 
minibatch_size = 32                         #use m for Batch Gradient Descent instead of Mini Batch Gradient Descent

optimizer = "adam"                          #gradient descent type ("gd" or "adam" available)
beta1 = 0.9                                 #momentum parameter for Adam Optimizer
beta2 = 0.999                               #momentum parameter for Adam Optimizer
epsilon=10e-8                               #for Adam Optimizer

parameters = tensor_model(X_train, Y_train, X_test, Y_test, layer_dims, optimizer, beta1,beta2,epsilon, learning_rate, num_epochs, minibatch_size)