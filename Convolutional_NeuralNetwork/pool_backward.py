import numpy as np
from create_mask_from_window import create_mask_from_window
from distribute_value import distribute_value

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    """

    (A_prev, hparameters) = cache                                       # Retrieve information from cache
    
    # Retrieve hyperparameters from "hparameters" 
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape
    m, n_H_prev, n_W_prev, n_C_prev = np.shape(A_prev)
    m, n_H, n_W, n_C = np.shape(dA)
    
    dA_prev = np.zeros(A_prev.shape)                                    # Initialize dA_prev with zeros
    
    for i in range(m):                                                  # loop over the training examples
        
        a_prev = A_prev[i]                                              # select training example from A_prev 
        
        for h in range(n_H):                                            # loop on the vertical axis
            for w in range(n_W):                                        # loop on the horizontal axis
                for c in range(n_C):                                    # loop over the channels (depth)
                    # Find the corners of the current "slice"
                    vert_start = stride*h
                    vert_end = stride*h + f
                    horiz_start = stride*w
                    horiz_end = stride*w + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                        # Create the mask from a_prev_slice
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i, h, w, c]*mask
                        
                    elif mode == "average":
                        
                        # Get the value a from dA
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf
                        shape = (f,f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
                        
    assert(dA_prev.shape == A_prev.shape)                                   # Making sure your output shape is correct
    
    return dA_prev