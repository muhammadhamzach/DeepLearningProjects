import tensorflow as tf

def create_placeholders(n_x, n_y):
    ##Creates the placeholders for the tensorflow session.

    X = tf.compat.v1.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.compat.v1.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y