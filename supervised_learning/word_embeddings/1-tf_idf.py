#!/usr/bin/env python3
'''
Creates a Bag of Words embedding matrix
'''

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    '''
    Creates a Bag of Words embedding matrix
    '''

    # Initialize the TfidfVectorizer with the given vocabulary if provided
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit the vectorizer on the sentences and transform the sentences to a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Convert the sparse matrix to a dense numpy array
    embeddings = tfidf_matrix.toarray()

    # Get the feature names
    features = vectorizer.get_feature_names_out()

    return embeddings, features
