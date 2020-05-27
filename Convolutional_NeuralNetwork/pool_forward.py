import numpy as np

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape            # Retrieve dimensions from the input shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))                            # Initialize output matrix A              

    for i in range(m):                                          # loop over the training examples
        for h in range(n_H):                                    # loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice" 
            vert_start = h*stride
            vert_end = h*stride + f
            
            for w in range(n_W):                                # loop on the horizontal axis of the output volume
                # Find the vertical start and end of the current "slice"
                horiz_start = w*stride
                horiz_end = w*stride + f
                
                for c in range (n_C):                           # loop over the channels of the output volume
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]
                    
                    # Compute the pooling operation on the slice. 
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)                               # Store the input and hparameters in "cache" for pool_backward
    
    assert(A.shape == (m, n_H, n_W, n_C))                       # Making sure your output shape is correct
    
    return A, cache