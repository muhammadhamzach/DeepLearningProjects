import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):

    ##Creates the placeholders for the tensorflow session.

    X = tf.compat.v1.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_y])
    
    return X, Y