import numpy as np
import matplotlib.pyplot as plt
from load_2D_dataset import load_2D_dataset
from L_layer_model import L_layer_model
from twoD_plot import twoD_plot

'exec(%matplotlib inline)'         #for Python IDE
'exec(%load_ext autoreload)'       #for Python IDE
'exec(%autoreload 2)'              #for Python IDE

plt.rcParams['figure.figsize'] = (7.0, 4.0)              #set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(3)

train_x, train_y, test_x, test_y = load_2D_dataset()   #train & test both available
m = train_x.shape[1]                                    

#setting up Neural Network Hyper-parameters parameters     
layers_dims = [train_x.shape[0], 20, 3, 1]              #for train/test dataset recommended layers_dims 
learning_rate = 0.3                                     #learning rate alpha
lambd = 0.7                                             #regularization parameter lambda
num_iterations = 10100                                  #no of iterations of code
keep_prob = 0.86                                        #drop out regularization parameter
optimizer = "gd"                                        #gradient descent optimization between parameters ("gd", "momentum, "adam")
epsilon = 10e-8                                         #epsilon constant used in adam gradient descent
mini_batch_size = m                                     #for batch gradient descent default it to variable m
beta1 = 0.9                                             #for velcoity optimization in gradient descent
beta2 = 0.999                                           #for adam optimization in gradient descent

print_cost = True                                       #printing cost after every 1000 iterations
#Training the algorithm
parameters = L_layer_model(train_x, train_y, test_x, test_y, layers_dims, mini_batch_size, lambd, learning_rate, beta1, beta2,num_iterations, keep_prob, epsilon, optimizer, print_cost)
twoD_plot(parameters, train_x, train_y)




