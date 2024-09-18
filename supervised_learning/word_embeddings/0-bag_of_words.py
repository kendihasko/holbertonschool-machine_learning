#!/usr/bin/env python3

import numpy as np


def bag_of_words(sentences, vocab=None):
    # Preprocess the sentences
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]
    
    # Create the vocabulary if none is provided
    if vocab is None:
        # Flatten the list of words and get unique words
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
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1  # Increment count for the word

    return embeddings, features

# Example usage:
sentences = ["This is a sentence", "This is another sentence", "And this is a third one"]
embeddings, features = bag_of_words(sentences)

print("Embeddings:\n", embeddings)
print("Features:\n", features)
