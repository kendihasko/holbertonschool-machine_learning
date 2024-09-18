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
    sentences: list of sentences to analyze
    vocab: list of vocabulary words to use for the analysis.
           If None, all words within sentences should be used.
           
    Returns:
    embeddings: numpy.ndarray of shape (s, f) containing the embeddings
                s is the number of sentences in sentences
                f is the number of features analyzed
    features: list of the features used for embeddings
    
    Note: The use of the gensim library is prohibited.
    """
    
    # Tokenize sentences into words, remove punctuation, and lowercasing
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
    
    # Return the embeddings and the features (vocabulary)
    return embeddings, vocab
