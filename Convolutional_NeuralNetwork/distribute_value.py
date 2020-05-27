import numpy as np

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    """
    
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix
    average = dz/ (n_H*n_W)
    
    # Create a matrix where every entry is the "average" value
    a = np.ones(shape)* average
        
    return a