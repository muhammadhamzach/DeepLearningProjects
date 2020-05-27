import numpy as np
from model import model

data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
chars = sorted(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

parameters = model(data, ix_to_char, char_to_ix)