import numpy as np
from L_model_forward import L_model_forward

def predict(X, parameters, keep_prob = 1):
    ## This function is used to predict the results of a  L-layer neural network
    p = np.zeros((1,X.shape[1]))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters, keep_prob)
  
    # convert probas to 0/1 predictions
    for a in range(0, probas.shape[1]):
        if probas[0,a] > 0.5:
            p[0,a] = 1
        else:
            p[0,a] = 0
         
    return p