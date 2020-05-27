from keras.models import Model
from keras.layers import Input, Lambda, RepeatVector
import keras.backend as K
import tensorflow as tf

def one_hot(x):
    x = K.argmax(x)
    x = tf.one_hot(x, 78) 
    x = RepeatVector(1)(x)
    return x

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0
    # Create an empty list of "outputs" to later store your predicted values
    outputs = []
    
    # Loop over Ty and generate a value at every time step
    for _ in range(Ty):
        
        # Perform one step of LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        #Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)
        # Append the prediction "out" to "outputs"
        outputs.append(out)
        
        # Select the next value according to "out",
        x = Lambda(one_hot)(out)
        
    # Create model instance with the correct "inputs" and "outputs"
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    return inference_model