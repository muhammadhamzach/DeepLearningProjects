import numpy as np
#from sigmoid import sigmoid

def sigmoid_backward(dA, cache):
    ## Implement the backward propagation for a single SIGMOID unit.

    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == cache.shape)
    
    return dZ
