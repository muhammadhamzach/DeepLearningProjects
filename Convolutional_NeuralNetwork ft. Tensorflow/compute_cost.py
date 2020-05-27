import tensorflow as tf

def compute_cost(Z3, Y):
    ##Computes the cost

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    
    return cost