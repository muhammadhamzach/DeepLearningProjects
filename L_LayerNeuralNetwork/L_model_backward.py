import numpy as np
from linear_activation_backward import linear_activation_backward
from numpy import *
#from linear_backward import linear_backward

def L_model_backward(AL, Y, caches, keep_prob, lambd):
    ## Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)     # Y is the same shape as AL
     
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dAL[isnan(dAL)]=0
     
    # Lth layer (SIGMOID -> LINEAR) gradients. 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, lambd,activation = "sigmoid")
    if keep_prob < 1:
        temp_cache = caches[L-2]
        linear_cache_temp, activation_cache_temp, D = temp_cache
        grads["dA" + str(L-1)] = D * grads["dA" + str(L-1)]
        grads["dA" + str(L-1)] = grads["dA" + str(L-1)]/keep_prob
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,lambd, activation = "relu")
        
        if keep_prob < 1 & l != 0:
            temp_cache = caches[l-1]
            linear_cache_temp, activation_cache_temp, D = temp_cache
            dA_prev_temp = D * dA_prev_temp
            dA_prev_temp = dA_prev_temp/keep_prob
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads