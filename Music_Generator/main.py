from __future__ import print_function
import sys
import numpy as np
from load_music_utils import load_music_utils
from generate_music import generate_music
from keras.layers import Dense, LSTM, Reshape
from keras.optimizers import Adam
from djmodel import djmodel
from music_inference_model import music_inference_model
from predict_and_sample import predict_and_sample

train_file = 'data/original_metheny.mid'
m = 60          #no of training examples       
Tx = 30         #length of input sequence
n_a = 64        #number of dimensions for the hidden state of each LSTM cell.
Ty = 50         #length of output sequence

X, Y, n_values, indices_values = load_music_utils(train_file, m, Tx)

#to be trained LSTM_cell & densor
reshapor = Reshape((1, n_values))       
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')

a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

#training on the given midi audio file
model = djmodel(reshapor, LSTM_cell, densor, Tx = Tx , n_a = n_a, n_values = n_values)
opt = Adam(lr=0.01, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X, a0, c0], list(Y), epochs=100)

#creating a new midi audio file
inference_model = music_inference_model(LSTM_cell, densor, n_values = n_values, n_a = n_a, Ty = Ty)
out_stream = generate_music(train_file, x_initializer, a_initializer, c_initializer,inference_model)