import numpy as np

def initialize_parameters(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(0, L):
        
        parameters['W' + str(l+1)] = np.random.randn(layer_dims[l])* np.sqrt(2/layer_dims[l-1]) 
        parameters['b' + str(l+1)] = np.zeros(([1,1,1,np.shape(layer_dims)[3]]))
        
        assert(parameters['W' + str(l+1)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l+1)].shape == (np.shape(layer_dims)[3]))

        
    return parameters