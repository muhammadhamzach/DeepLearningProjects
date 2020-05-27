import tensorflow as tf
import numpy as np


def initialize_parameters(layer_dims):
    ## Initializes parameters to build a neural network with tensorflow. The shapes are:

    tf.compat.v1.random.set_random_seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        parameters['W' + str(l)] = tf.compat.v1.get_variable('W' + str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.keras.initializers.GlorotUniform(seed = 1))
        parameters['b' + str(l)] = tf.compat.v1.get_variable('b' + str(l), [layer_dims[l], 1], initializer = tf.zeros_initializer())
        np.zeros((layer_dims[l], 1))
    
    return parameters