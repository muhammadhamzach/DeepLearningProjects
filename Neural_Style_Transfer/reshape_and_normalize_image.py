import os
import sys
from PIL import Image
import numpy as np

class CONFIG:
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 


def reshape_and_normalize_image(image):
    """
    Reshape and normalize the input image (content or style)
    """
    # Reshape image to mach expected input of VGG16
    image = np.reshape(image, ((1,) + image.shape))
    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS
    
    return image