from Neural_Style import Neural_Style_Image

content_image_path = "images/content.jpeg"
style_image_path = "images/sandstone.jpg"
model_path = "pre-trained_model/"
output_image_path = "output/"

num_iterations = 3000
alpha = 1e4                  #hyper-paramter giving weight to content_imgae cost
beta = 1e-1                   #hyperparameter giving weight to style_image cost
learning_rate = 0.02          #learning rate for gradient descent

#further parameters to be modified can be found in Neural_Style.py
Neural_Style_Image(content_image_path, style_image_path, output_image_path, model_path, num_iterations, alpha, beta, learning_rate)