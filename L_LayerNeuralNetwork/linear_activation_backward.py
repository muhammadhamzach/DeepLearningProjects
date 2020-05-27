from relu_backward import relu_backward
from sigmoid_backward import sigmoid_backward
from linear_backward import linear_backward

def linear_activation_backward(dA, cache,  lambd, activation):
    ##  Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    linear_cache, activation_cache, D = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, lambd, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, lambd, linear_cache)

    
    return dA_prev, dW, db