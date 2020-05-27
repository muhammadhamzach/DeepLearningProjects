import os
import sys
import scipy.io
import scipy.misc
import imageio
from PIL import Image
import numpy as np

class CONFIG:
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 

def save_image(path, image):
    
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS
    
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    imageio.imwrite(path, image)