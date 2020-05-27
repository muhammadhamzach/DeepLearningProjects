import numpy as np
from conv_single_step import conv_single_step
from zero_pad import zero_pad

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    """
    
    (m, n_H_prev, n_W_prev, _) = np.shape(A_prev)

    (f, f, _, n_C) = np.shape(W)                 # Retrieve dimensions from W's shape
    
    # Retrieve information from "hparameters" 
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume
    n_H = int((n_H_prev-f+2*pad)/stride) + 1
    n_W = int((n_W_prev-f+2*pad)/stride) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))                    # Initialize the output volume Z with zeros.
    
    A_prev_pad = zero_pad(A_prev, pad)                  # Create A_prev_pad by padding A_prev
    
    for i in range(m):                                  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]                      # Select ith training example's padded activation
        for h in range(n_H):                            # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice" 
            vert_start = h*stride
            vert_end = h*stride + f
            
            for w in range(n_W):                        # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current 
                horiz_start = w*stride
                horiz_end = w*stride + f
                
                for c in range(n_C):                    # loop over channels (= #filters) of the output volume
                                        
                    # Use the corners to define the (3D) slice of a_prev_pad 
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        

    
    assert(Z.shape == (m, n_H, n_W, n_C))               # Making sure your output shape is correct
    
    
    cache = (A_prev, W, b, hparameters)                 # Save information in "cache" for the backprop
    
    return Z, cache