import numpy as np

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    """
    mask = (x == np.max(x))

    return mask