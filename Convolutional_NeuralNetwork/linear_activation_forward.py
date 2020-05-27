from linear_forward import linear_forward
from sigmoid import sigmoid
from relu import relu
import numpy as np

def linear_activation_forward(A_prev, n_y):
    ## Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    layers_dims = [np.shape(A_prev)[0], n_y]
    parameters = {}
    parameters['W' + str(l+1)] = np.random.randn(layer_dims)* np.sqrt(2/layer_dims[l-1]) 
    parameters['b' + str(l+1)] = np.zeros(())
    Z = np.dot(W, A) + b

    Z, linear_cache = linear_forward(A_prev, W, b)
    
         
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache, D)

    return A, cache