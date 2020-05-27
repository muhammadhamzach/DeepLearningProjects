import tensorflow as tf
from tensorflow.python.framework import ops
tf.compat.v1.disable_eager_execution()
from create_placeholders import create_placeholders
from initialize_parameters import initialize_parameters
from forward_propagation import forward_propagation
from compute_cost import compute_cost
from random_mini_batches import random_mini_batches
import matplotlib.pyplot as plt
import numpy as np


def model(X_train, Y_train, X_test, Y_test, ksize, layers_dims, optimizer="adam", learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True, beta1=0.9, beta2=0.999, epsilon=1e-8, 
          strides=[1,1,1,1]):
    """
    Implements a L-layer ConvNet in Tensorflow:
    (CONV2D -> RELU -> MAXPOOL)^L -> FLATTEN -> FULLYCONNECTED
    """

    ops.reset_default_graph()                           # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.random.set_random_seed(1)              # to keep results consistent (tensorflow seed)
    seed = 3                                            # to keep results consistent (numpy seed)
    np.shape(X_train)
    (m, n_H0, n_W0, n_C0) = np.shape(X_train)             
    n_y = Y_train.shape[1]                            
    costs = []                                          # To keep track of the cost
    
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)   # Create Placeholders of the correct shape
   
    parameters = initialize_parameters(layers_dims)                 # Initialize parameters

    Z = forward_propagation(X, parameters, ksize, strides, n_y)     # Forward propagation
   
    cost = compute_cost(Z, Y)                          # Cost function
    
    # Backpropagation: Define the tensorflow optimizer
    if optimizer == "gd":
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    elif optimizer == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate, beta1=beta1, beta2=beta2, epsilon = epsilon).minimize(cost)

    init = tf.compat.v1.global_variables_initializer()  # Initialize all the variables globally
     
    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:
        
        sess.run(init)                                  # Run the initialization
        
        for epoch in range(num_epochs):                 # Do the training loop

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)   # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                
                (minibatch_X, minibatch_Y) = minibatch  # Select a minibatch

                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
              
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return parameters