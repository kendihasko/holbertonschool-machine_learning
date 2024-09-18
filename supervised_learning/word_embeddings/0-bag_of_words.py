#!/usr/bin/env python3
'''
Creates a Bag of Words embedding matrix
'''

import numpy as np
from collections import Counter
import string

def preprocess(sentence, normalization_dict=None):
    """Helper function to lowercase, remove punctuation, and normalize word variations."""
    # Convert to lowercase
    sentence = sentence.lower()
    
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize word variations
    if normalization_dict:
        words = sentence.split()
        normalized_words = [normalization_dict.get(word, word) for word in words]
        sentence = ' '.join(normalized_words)
    
    return sentence

def bag_of_words(sentences, vocab=None, normalization_dict=None):
    # Step 1: Preprocess sentences (lowercase, remove punctuation, handle variations)
    sentences = [preprocess(sentence, normalization_dict) for sentence in sentences]
    
    # Step 2: Tokenize the sentences
    tokenized_sentences = [sentence.split() for sentence in sentences]

    # Step 3: Create or use the provided vocabulary
    if vocab is None:
        # Create vocabulary from unique words, without sorting by frequency
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    else:
        vocab = sorted(vocab)  # Sort vocab alphabetically if provided

    # Step 4: Count s (number of sentences) and f (number of unique features/words)
    s = len(sentences)  # Number of sentences
    f = len(vocab)      # Number of features (words in the vocab)

    # Step 5: Initialize the Bag of Words matrix (embeddings)
    embeddings = np.zeros((s, f), dtype=int)  # s x f matrix initialized to zeros

    # Step 6: Populate the matrix with word counts
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)  # Count the occurrences of each word in the sentence
        for j, word in enumerate(vocab):
            embeddings[i, j] = word_counts[word]  # Set the count of the word in the BoW matrix

    return embeddings, vocab  # Return the embeddings and the list of features (vocabulary)

# Define normalization dictionary
normalization_dict = {
    'childrens': 'children',
    # Add other word variations here if needed
}

# Example usage
sentences = [
    "The children are playing in the park.",
    "The childrens toys were scattered all over."
]

embeddings, features = bag_of_words(sentences, normalization_dict=normalization_dict)
print("Bag of Words embeddings:\n", embeddings)
print("\nFeatures (vocabulary):", features)
