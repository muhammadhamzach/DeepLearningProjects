from preprocess import get_musical_data,get_corpus_data
from music_utils import data_processing

def load_music_utils(train_file, m, Tx):
    _, abstract_grammars = get_musical_data(train_file)
    corpus, _, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, m, Tx)   
    return (X, Y, N_tones, indices_tones)