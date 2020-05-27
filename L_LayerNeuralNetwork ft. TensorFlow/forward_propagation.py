import tensorflow as tf


def forward_propagation(X, parameters):
    ##Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    A = X
    L = len(parameters) // 2  
    
    tf.compat.v1.random.set_random_seed(1)
      
    for l in range(1, L+1):     
        Z = tf.add(tf.matmul(parameters['W'+str(l)],A),parameters['b'+str(l)])
        if l != L:
            A = tf.nn.relu(Z)
 
    return Z