import numpy as np

def relu(Z):
    ## Implement the RELU function.
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache