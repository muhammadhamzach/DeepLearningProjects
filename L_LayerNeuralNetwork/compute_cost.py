import numpy as np

def compute_cost(AL, Y, parameters, lambd):
    ## Implement the cost function
    
    #m = Y.shape[1]
    L = len(parameters) // 2 

    cost = np.nansum(np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y))
    #cost = 1./m *(logprobs)
    

    reg_cost = 0
    for l in range(L):
        reg_cost = reg_cost + np.sum(np.square(parameters["W" + str(l+1)]))

    cost = cost + (lambd/(2))*reg_cost
    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost