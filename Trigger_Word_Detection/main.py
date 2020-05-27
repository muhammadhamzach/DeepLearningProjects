from load_raw_audio import load_raw_audio
from create_training_example import create_training_example
from model import model
from keras.models import load_model
from keras.optimizers import Adam
from detect_triggerword import detect_triggerword
from chime_on_activate import chime_on_activate
from preprocess_audio import preprocess_audio
import matplotlib.pyplot as plt
import numpy as np

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

"""
                    # CODE FOR CREATING A TRAINING EXAMPLE
        #Activations & Negatives will be super-imposed on the Background file
Ty = 1375 # The number of time steps in the output of our model
activates, negatives, backgrounds = load_raw_audio() #loading data frommthe raw_data folder
X, Y = create_training_example(background, activates, negatives, Ty)
"""

#Pre-Trained Examples for the "activate" as an action word
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

model = model(input_shape = (Tx, n_freq))
# pre-trained model for the given activation word with 4000 training example
# if different action word to be used then this same code can be re-run m-times to get the desired model
model = load_model('./models/tr_model.h5')
opt = Adam(lr=0.0001, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, batch_size = 5, epochs=1)

#testing a file, some examples are given in the audio_examples folder
filename = "audio_examples/my_audio.wav"
preprocess_audio(filename)
chime_threshold = 0.5
prediction = detect_triggerword(filename, model)
chime_on_activate(filename, prediction, chime_threshold)
plt.show()