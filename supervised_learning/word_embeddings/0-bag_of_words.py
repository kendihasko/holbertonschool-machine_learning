#!/usr/bin/env python3
'''
Creates a Bag of Words embedding matrix
'''

import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_text(text):
    '''
    Convert text to lowercase and replace punctuation with spaces.
    '''
    text = text.lower()
    text = re.sub(r'[{}]+'.format(re.escape(string.punctuation)), ' ', text)
    return text


def bag_of_words(sentences, vocab=None):
    '''
    Creates a Bag of Words embedding matrix
    '''
    # Create a CountVectorizer instance with custom preprocessing
    vectorizer = CountVectorizer(
        preprocessor=preprocess_text, vocabulary=vocab
        )

    # Fit and transform the sentences
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # Get the feature names (vocabulary)
    features = vectorizer.get_feature_names_out()

    # Format the features list
    formatted_features = ' '.join(f"'{item}'" for item in features)
    formatted_features = f"[{formatted_features}]"

    return embeddings, formatted_features
