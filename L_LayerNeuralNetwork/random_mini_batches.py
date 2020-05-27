import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    #Creates a list of random minibatches from (X, Y)
    
    m = X.shape[1]
    if mini_batch_size == m:
        mini_batch = (X, Y)
        return mini_batch
    else:
        np.random.seed(seed)
        
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))
    
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            end = m - mini_batch_size * math.floor(m / mini_batch_size)
            mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        return mini_batches