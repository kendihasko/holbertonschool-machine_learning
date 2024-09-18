#!/usr/bin/env python3

import numpy as np
import string

def remove_punctuation(word):
    """
    Remove punctuation from a word and return the cleaned word.
    """
    return word.translate(str.maketrans('', '', string.punctuation))

def normalize_word(word):
    """
    Normalize words to handle variations (e.g., 'childrens' -> 'children').
    """
    # Example normalization: handle plural forms or other variations here
    if word.endswith('s') and word[:-1] in {'children'}:
        return 'children'
    return word

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
        A formatted string representing the embeddings matrix with rows enclosed in square brackets.
    features_str: str
        A formatted string of features (vocabulary) enclosed in square brackets.
    """
    
    # Tokenize sentences into words, remove punctuation, normalize, and lowercase
    tokenized_sentences = [
        [normalize_word(remove_punctuation(word.lower())) for word in sentence.split()] 
        for sentence in sentences
    ]
    
    # If no vocabulary is provided, generate one from the sentences
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    # Ensure vocabulary order matches the output format
    vocab = sorted(vocab)
    
    # Create an empty matrix for the bag-of-words representation
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    # Create a mapping of vocabulary words to indices
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # Fill the matrix with word counts
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1
    
    # Format the embeddings matrix with each row enclosed in square brackets
    embeddings_str = "[\n" + "\n".join(f"[{' '.join(map(str, row))}]" for row in embeddings) + "\n]"
    
    # Format the features list with each word enclosed in single quotes and the whole list enclosed in square brackets
    features_str = "[{}]".format(" ".join(f"'{word}'" for word in vocab))
    
    return embeddings_str, features_str
