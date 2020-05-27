from model import model
from load_dataset import load_dataset
import numpy as np

X_train, Y_train, X_test, Y_test, classes = load_dataset()

layers_dims = [[4,4,3,8],[2,2,8,16]]        #dimesions of the convolutional neural network layer
learning_rate = 0.009                       #alpha learning rate of the system
num_epochs = 100                            #number of iterations 
minibatch_size = 64                         #use m for Batch Gradient Descent instead of Mini Batch Gradient Descent

optimizer = "adam"                          #gradient descent type ("gd" or "adam" available)
beta1 = 0.9                                 #momentum parameter for Adam Optimizer
beta2 = 0.999                               #momentum parameter for Adam Optimizer
epsilon=1e-8                                #for Adam Optimizer

strides=[1,1,1,1]                           #for convolution and max_polling             
ksize=[[1,8,8,1],[1,4,4,1]]                 #for max_polling

"""parameters = model(X_train, Y_train, X_test, Y_test, , ksize, layers_dims, optimizer, learning_rate, 
                    num_epochs, minibatch_size, beta1, beta2, epsilon, strides)""" ##model function argument
parameters = model(X_train, Y_train, X_test, Y_test, ksize=ksize, layers_dims=layers_dims)