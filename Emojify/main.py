import numpy as np
from emo_utils import read_csv, read_glove_vecs, convert_to_one_hot
import emoji
from Emojify_V2 import Emojify
from sentences_to_indices import sentences_to_indices
from label_to_emoji import label_to_emoji
np.random.seed(0)

X_train, Y_train = read_csv('data/train_emoji.csv')

maxLen = len(max(X_train, key=len).split())
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')

model = Emojify((maxLen,), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)

sentence_test ='I love you' 
x_test = np.array([sentence_test])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))