# Neural Style Learning

* main.py is used to execute the program
* User only needs to input the content image and style image path
* The training model used here is imagenet-vgg-verydeep-19.mat (This model needs to be downloaded and placed in the pre-trained_model folder, link in the specfic folder)
* User can also specify the number of iterations the code will run
* Detail modifications to hyper-parameters can also be done
* Scripts use Tensorflow 2.x
* A sample video made from this script is also added. It is made from 500 images over 10000 iterations.
* The content image and style image can be of different dimensions but in pre-processing style image will take the shape of content image

## Note
* A Jupyter Notebook is also provided to play with this program (for that notebook all these python files in repository are un-necessary and it can work standalone i.e it only needs the vgg-model link and the content/style image paths)


## Reference
* This script is partially derived from the deeplearning.ai Convolutional Network course 