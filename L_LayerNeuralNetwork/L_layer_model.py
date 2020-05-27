import numpy as np
import matplotlib.pyplot as plt
from initialize_parameters_deep import initialize_parameters_deep
from L_model_forward import L_model_forward
from compute_cost import compute_cost
from L_model_backward import L_model_backward
from update_parameters import update_parameters
from optimization import optimization
from initilization import initilization
from random_mini_batches import random_mini_batches
from predict import predict
from twoD_plot import twoD_plot


def L_layer_model(X, Y, test_x, test_y, layers_dims, mini_batch_size, lambd = 0, learning_rate = 0.0075, beta1=0.9, beta2=0.999, num_iterations = 3000, keep_prob = 1, epsilon = 10e-8, optimizer = "gd", print_cost=False):
    ##  Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    np.random.seed(3)
    costs = []
    m = X.shape[1] 
    seed = 10
    t = 0                            # initializing the counter required for Adam update
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    v,s = initilization(parameters)
    
    
    # Loop (gradient descent)
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
            AL, caches = L_model_forward(minibatch_X, parameters, keep_prob)

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

    #plot the cost
    plt.figure(2)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    #Training and Test Set Accuracy Measure
    pred_train = predict(X, parameters)
    print("Training Accuracy: "  + str(np.sum((pred_train == Y)/X.shape[1])*100) + " %")
    pred_test = predict(test_x, parameters) 
    print("Test Accuracy: "  + str(np.sum((pred_test == test_y)/test_x.shape[1])*100) + " %")

    #plotting the 2D dataset with solved parameters
    #twoD_plot(parameters, X, Y)  
    
    dreturn parameters