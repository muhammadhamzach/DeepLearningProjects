from relu import relu
import numpy as np
from conv_forward import conv_forward
from pool_forward import pool_forward
from L_model_forward import L_model_forward

def conv_model_forward(X,  parameters, hyperparameters,mode, n_y):
    ## Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    caches = []
    A = X
    L = len(parameters) // 2  
    np.random.seed(1)
      
    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(0, L):      
        A_prev = A 
        Z, conv_cache = conv_forward(A_prev, parameters['W'+str(l+1)], parameters['b'+str(l)], hyperparameters)
        A_z = np.maximum(0,Z)           #relu
        A, pool_cache = pool_forward(A_z, hyperparameters, mode)
        caches.append(conv_cache, pool_cache)
    
    # Implement LINEAR -> SIGMOID.
    #keep_prob =1
    #AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+ str(L)],keep_prob,activation = "sigmoid")
    #caches.append(cache)
    A = A.reshape(np.shape(A)[0],-1).T
    AL, caches = linear_activation_forward(A,n_y)
    #assert(AL.shape == (1,X.shape[1]))
      
    #return AL, caches