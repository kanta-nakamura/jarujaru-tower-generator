import pickle
import numpy as np
from pathlib import Path

vocab_file = 'vocab.pkl'
corpus_file = 'corpus.npy'
dataset_dir = Path(__file__).resolve().parents[2] / 'data/processed'

def load_data():

    corpus_path = dataset_dir / corpus_file
    vocab_path = dataset_dir / vocab_file

    corpus = np.load(corpus_path)

    with open(vocab_path, 'rb') as f:
        word_to_id, id_to_word = pickle.load(f)

    return corpus, word_to_id, id_to_word