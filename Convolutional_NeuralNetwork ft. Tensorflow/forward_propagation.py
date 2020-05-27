import tensorflow as tf
import numpy as np
#from keras.layers import Input, Flatten

def forward_propagation(X, parameters, ksize, strides, n_y):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    """
    L = len(parameters)
    
    P = X
    for l in range(0,L): 
        Z = tf.nn.conv2d(P ,parameters['W'+str(l+1)], strides = strides, padding = 'SAME')
        A = tf.nn.relu(Z)
        P = tf.nn.max_pool(A, ksize = ksize[l], strides = ksize[l], padding = 'SAME')

    F = tf.keras.layers.Flatten()(P)
    Z = tf.keras.layers.Dense(units=n_y)(F)
    return Z