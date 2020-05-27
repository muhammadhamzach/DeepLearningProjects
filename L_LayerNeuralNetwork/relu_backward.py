import numpy as np
from relu import relu

def relu_backward(dA, cache):
    ## Implement the backward propagation for a single RELU unit.

    Z = cache
    dZ = np.array(dA, copy=True)  
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ