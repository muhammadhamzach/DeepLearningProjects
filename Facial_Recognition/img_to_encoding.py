import numpy as np
import os
import cv2
from keras.models import Model

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img1 = cv2.resize(img1,(96,96))
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding