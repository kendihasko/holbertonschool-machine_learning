#!/usr/bin/env python3

import numpy as np

def bag_of_words(sentences, vocab=None):
    # Preprocess the sentences
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    
    # Create the vocabulary if none is provided
    if vocab is None:
        vocab = set(word for sentence in tokenized_sentences for word in sentence)
    
    # Convert vocabulary to a sorted list
    features = sorted(vocab)
    
    # Initialize the embedding matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    # Create a mapping of words to their index in the features list
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Populate the embedding matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            # Remove punctuation from the word
            word = word.strip("!.,'\"")
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1  # Increment count for the word

    return embeddings, features
