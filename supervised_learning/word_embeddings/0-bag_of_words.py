#!/usr/bin/env python3
'''
Creates a Bag of Words embedding matrix
'''

import numpy as np
import string
from collections import Counter

def preprocess_text(text):
    """Convert text to lowercase, replace punctuation with spaces."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text

def remove_trailing_s(word, vocab_set):
    """Remove trailing 's' from a word if it exists and the word is in the vocabulary."""
    if word.endswith('s') and word[:-1] in vocab_set:
        return word[:-1]
    return word

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
    
    # Create a set of vocabulary words for quick lookup
    vocab_set = set(features)
    
    # Create a word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Initialize the embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    # Fill the embeddings matrix
    for i, sentence in enumerate(processed_sentences):
        word_counts = Counter(sentence.split())
        for word, count in word_counts.items():
            word = remove_trailing_s(word, vocab_set)
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count
    
    # Format the features list
    formatted_features = ' '.join(f"'{item}'" for item in features)
    formatted_features = f"[{formatted_features}]"
    
    return embeddings, formatted_features
