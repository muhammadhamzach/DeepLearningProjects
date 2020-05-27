import numpy as np

def sigmoid(Z):

    ## Implements the sigmoid activation in numpy
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache