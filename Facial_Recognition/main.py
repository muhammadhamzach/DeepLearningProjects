import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from faceRecoModel import faceRecoModel
from triplet_loss import triplet_loss
from load_weight import load_weights_from_FaceNet
from verify import facial_verification
from database_entry import database_entry
from who_is_it import facial_recognition

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = database_entry(FRmodel)

picture_location = "images/m.jpg"
person_name = ""                #optional, if no name then use "" 

if person_name == "":
    facial_recognition(picture_location, database, FRmodel)
else:
    facial_verification(picture_location, person_name, database, FRmodel)

