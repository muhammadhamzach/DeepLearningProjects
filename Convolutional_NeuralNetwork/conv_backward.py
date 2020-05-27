import numpy as np
from zero_pad import zero_pad

def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    """
 
    (A_prev, W, b, hparameters) = cache                             # Retrieve information from "cache"
    
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)            # Retrieve dimensions from A_prev's shape
    
    (f, f, n_C_prev, n_C) = np.shape(W)                             # Retrieve dimensions from W's shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m, n_H, n_W, n_C) = np.shape(dZ)                               # Retrieve dimensions from dZ's shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pad A_prev and dA_prev
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                                              # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                                        # loop over vertical axis of the output volume
            for w in range(n_W):                                    # loop over horizontal axis of the output volume
                for c in range(n_C):                                # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        
        dA_prev[i, :, :, :] =da_prev_pad[pad:-pad, pad:-pad, :]     # Set the ith training example's dA_prev to the unpadded da_prev_pad
    
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))      # Making sure your output shape is correct
    
    return dA_prev, dW, db