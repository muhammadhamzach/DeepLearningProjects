# Facial Verification & Facial Recognition

* This repository contains the requisite files for both facial verification and facial recognition made using inception model
* images folder contains all the pictures that are needed to be stored in database by user
* in order to add an image into the facial database, put the image in images folders and then open up the database.py file and add the details about the corresponding pic in the database by using the following code format  
database["name of person"] = img_to_encoding("location of the image", FRmodel)
* if you want to re-create the model from scratch then open up the main.py file and it will run all the files and also make a facial_mode.h5 for later use
* In main.py you can specify the path of the image to be recognized and also the name of person too but name one is optional (if no name then put "" in front of name)
* If you want to use the pre-made model and save time then run the pre_loaded_model.py file and put the path of the image and the name (optional) there

### Reference:
* This code is based on the Convolutional Neural Networks course assignment on coursera and also simplified from the github project on https://github.com/iwantooxxoox/Keras-OpenFace/