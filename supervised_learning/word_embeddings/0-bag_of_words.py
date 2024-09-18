#!/usr/bin/env python3

import numpy as np
from keras.preprocessing.text import Tokenizer

def bag_of_words(sentences, vocab=None):
    # Initialize the Tokenizer
    tokenizer = Tokenizer()

    # Fit the tokenizer on the sentences
    tokenizer.fit_on_texts(sentences)

    # Convert sentences to sequences
    sequences = tokenizer.texts_to_sequences(sentences)

    # Create the bag of words embedding matrix
    embeddings = np.zeros((len(sentences), len(tokenizer.word_index) + 1), dtype=int)

    for i, seq in enumerate(sequences):
        for word_index in seq:
            embeddings[i, word_index] += 1  # Count the occurrences of each word

    # Features are the words in the tokenizer's index
    features = sorted(tokenizer.word_index.keys())

    return embeddings, features
