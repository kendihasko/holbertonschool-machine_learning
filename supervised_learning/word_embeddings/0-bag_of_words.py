#!/usr/bin/env python3
'''
Creates a Bag of Words embedding matrix
'''

import numpy as np
import string
from collections import Counter

def preprocess_text(text):
    """Convert text to lowercase and replace punctuation with spaces."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text

def bag_of_words(sentences, vocab=None):
    # Preprocess sentences
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    
    # Create a vocabulary if not provided
    if vocab is None:
        # Tokenize sentences and create a vocabulary from all words
        words = [word for sentence in processed_sentences for word in sentence.split()]
        features = sorted(set(words))
    else:
        features = vocab
    
    # Create a word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Initialize the embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    # Fill the embeddings matrix
    for i, sentence in enumerate(processed_sentences):
        word_counts = Counter(sentence.split())
        for word, count in word_counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count
    
    return embeddings, features
