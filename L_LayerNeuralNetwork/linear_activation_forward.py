from linear_forward import linear_forward
from sigmoid import sigmoid
from relu import relu
import numpy as np

def linear_activation_forward(A_prev, W, b,keep_prob, activation):
    ## Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":   
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        A, activation_cache = relu(Z)
        
    D = np.random.rand(A.shape[0],A.shape[1]) 
    if keep_prob < 1:
        D = (D < keep_prob).astype(int) 
        A = A * D                                      
        A = A/keep_prob
         
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache, D)

    return A, cache