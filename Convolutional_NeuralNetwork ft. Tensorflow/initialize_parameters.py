import tensorflow as tf
import numpy as np


def initialize_parameters(layers_dims):
    ## Initializes parameters to build a neural network with tensorflow. The shapes are:

    tf.compat.v1.random.set_random_seed(1)
    parameters = {}
    
    L = len(layers_dims)

    for l in range(0, L):
        parameters['W' + str(l+1)] = tf.compat.v1.get_variable('W' + str(l+1), layers_dims[l], initializer =tf.keras.initializers.GlorotUniform(seed=0))

    return parameters