import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
from verify import facial_verification
from database_entry import database_entry
from who_is_it import facial_recognition

with CustomObjectScope({'tf': tf}):
    model = load_model('./facial_model.h5', compile=False)

database = database_entry(model)

picture_location = "images/m.jpg"
person_name = ""                        #optional, if no name then use "" 

if person_name == "":
    facial_recognition(picture_location, database, model)
else:
    facial_verification(picture_location, person_name, database, model)