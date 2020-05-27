import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import scipy.io
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from model_nn import model_nn
from compute_style_cost import compute_style_cost
from total_cost import total_cost
from compute_content_cost import compute_content_cost
from load_vgg_model import load_vgg_model
from generate_noise_image import generate_noise_image
from reshape_and_normalize_image import reshape_and_normalize_image
from resizing import resizing


def Neural_Style_Image(content_image_path, style_image_path,output_image_path, model_path,num_iterations = 100, alpha = 10, beta = 40, learning_rate=0.02):

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()

    resizing(content_image_path, style_image_path)
    content_image = imageio.imread(content_image_path)
    content_image = reshape_and_normalize_image(content_image)

    style_image = imageio.imread("resized_style_image.png")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    model = load_vgg_model(model_path +"imagenet-vgg-verydeep-19.mat", content_image_path)

    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    STYLE_LAYERS = [
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0)]

    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(sess, model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha, beta)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(J)

    
    model_nn(sess, generated_image, model, train_step, J, J_content, J_style, output_image_path, num_iterations)