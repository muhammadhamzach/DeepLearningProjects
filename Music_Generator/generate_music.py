from keras.utils import to_categorical
from predict_and_sample import predict_and_sample
import numpy as np
from preprocess import get_corpus_data, get_musical_data
from music21 import stream, instrument, key, tempo, meter, note, converter,midi 
from qa import prune_grammar, prune_notes, clean_up_notes
from grammar import unparse_grammar

def generate_music(train_file, x_initializer, a_initializer, c_initializer, inference_model, T_y = 10, max_tries = 1000, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
    to save the music and play it.
    
    Arguments:
    model -- Keras model Instance, output of djmodel()
    corpus -- musical corpus, list of 193 tones as strings (ex: 'C,0.333,<P1,d-5>')
    abstract_grammars -- list of grammars, on element can be: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    tones -- set of unique tones, ex: 'A,0.250,<M2,d-4>' is one element of the set.
    tones_indices -- a python dictionary mapping unique tone (ex: A,0.250,< m2,P-4 >) into their corresponding indices (0-77)
    indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    Tx -- integer, number of time-steps used at training time
    temperature -- scalar value, defines how conservative/creative the model is when generating music
    
    Returns:
    predicted_tones -- python list containing predicted tones
    """
    chords, abstract_grammars = get_musical_data(train_file)
    _, _, _, indices_tones = get_corpus_data(abstract_grammars)
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    

    return out_stream
