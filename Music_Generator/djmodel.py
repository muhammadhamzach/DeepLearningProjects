from keras.models import Model
from keras.layers import Input,Lambda

def djmodel(reshapor,LSTM_cell,densor,Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0 using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    #Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)
        # Use reshapor to reshape x to be (1, n_values) 
        x = reshapor(x)
        # Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # add the output to "outputs"
        outputs.append(out)
        
    # Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    return model