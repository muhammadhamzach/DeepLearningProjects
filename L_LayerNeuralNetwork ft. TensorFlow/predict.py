import tensorflow as tf
from forward_propagation import forward_propagation


def predict(X, parameters):
     
    params = {}
    L = len(parameters) // 2 
    
    for l in range(1, L):    
        params['W' + str(l)] = tf.convert_to_tensor(parameters['W' + str(l)])
        params['b' + str(l)] = tf.convert_to_tensor(parameters['b' + str(l)])
    
    x = tf.placeholder("float", [X.shape[0], 1])
    
    Z = forward_propagation(x, params)
    p = tf.argmax(Z)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction