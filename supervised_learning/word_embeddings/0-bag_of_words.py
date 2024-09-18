#!/usr/bin/env python3

import numpy as np
import string

def remove_punctuation(word):
    """
    Remove punctuation from a word and return the cleaned word.
    """
    return word.translate(str.maketrans('', '', string.punctuation))

def bag_of_words(sentences, vocab=None):
    """
    Create a bag-of-words embedding matrix for the given sentences.

    sentences: list of str
        A list of sentences to analyze.

    vocab: list of str or None
        A list of vocabulary words to use for the analysis.
        If None, all words within sentences should be used.

    Returns:
    embeddings_str: str
        A formatted string representing the embeddings matrix with rows separated by newlines.
    features_str: str
        A formatted string of features (vocabulary) separated by spaces, enclosed in brackets.
    """
    
    # Tokenize sentences into words, remove punctuation, and lowercase
    tokenized_sentences = [
        [remove_punctuation(word.lower()) for word in sentence.split()] 
        for sentence in sentences
    ]
    
    # If no vocabulary is provided, generate one from the sentences
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    # Create an empty matrix for the bag-of-words representation
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    # Create a mapping of vocabulary words to indices
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # Fill the matrix with word counts
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1
    
    # Format the embeddings matrix with square brackets
    embeddings_str = "\n".join(f"[{' '.join(map(str, row))}]" for row in embeddings)
    
    # Format the features list with square brackets
    features_str = f"[{' '.join(vocab)}]"
    
    return embeddings_str, features_str
