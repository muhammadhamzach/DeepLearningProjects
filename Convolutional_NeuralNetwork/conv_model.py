import numpy as np
import matplotlib.pyplot as plt
from initialize_parameters import initialize_parameters
from conv_model_forward import conv_model_forward

def conv_model(X_test, Y_test, X_train, Y_train, layers_dims)

    np.random.seed(3)
    costs = []
    m = X.shape[1] 
    seed = 10

    parameters = initialize_parameters(layers_dims)
    v,s = initilization(parameters)
    for i in range(0, num_iterations):
        
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        
        cost_total = 0
        for minibatch in minibatches:
            # Select a minibatch
            if mini_batch_size == m:
                minibatch_X = X
                minibatch_Y = Y
            else:
                (minibatch_X, minibatch_Y) = minibatch
            
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = conv_model_forward(minibatch_X, parameters, keep_prob)

            # Compute cost.
            cost_total = cost_total + compute_cost(AL, minibatch_Y, parameters, lambd) 
    
            # Backward propagation
            grads = L_model_backward(AL, minibatch_Y, caches, keep_prob, lambd)
            
            t = t + 1 # Adam counter
            v,s,v_corrected, s_corrected = optimization(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
    
            # Update parameters.
            parameters = update_parameters(parameters, grads, learning_rate, v, v_corrected, s_corrected, epsilon, optimizer)          
            
            if mini_batch_size == m:
                break
            
        cost_avg = cost_total / m

        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 1000 == 0:
                costs.append(cost_avg)


