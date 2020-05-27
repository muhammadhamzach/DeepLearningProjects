from linear_activation_forward import linear_activation_forward
import numpy as np

def L_model_forward(X,  parameters, keep_prob):
    ## Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    
    caches = []
    A = X
    L = len(parameters) // 2  
    np.random.seed(1)
      
    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):      
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], keep_prob,activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID.
    keep_prob =1
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+ str(L)],keep_prob,activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
      
    return AL, caches