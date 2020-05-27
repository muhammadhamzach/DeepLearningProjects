from __future__ import print_function
from build_data import build_data
from generate_output import generate_output
from vectorization import vectorization
from keras.models import Model, load_model
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import io
import os
import sys


print("Loading text data...")
text = io.open('shakespeare.txt', encoding='utf-8').read().lower()

Tx = 40
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print("Creating training set...")
X, Y = build_data(text, Tx, stride = 3)
print("Vectorizing training set...")
x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices)


print("Loading model...")
model = load_model('./model/model_shakespeare_kiank.h5')
#print('Build model...')
#model = Sequential()
#model.add(LSTM(128, input_shape=(Tx, len(chars))))
#model.add(Dense(len(chars), activation='softmax'))
#optimizer = RMSprop(learning_rate=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)


print_callback = LambdaCallback()
model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])

generate_output(Tx,chars,char_indices, model,indices_char)