import numpy as np

def linear_forward(A, W, b):
    ## Implement the linear part of a layer's forward propagation.

    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache