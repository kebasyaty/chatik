# -*- coding: utf-8 -*-

# Machine learning.
# https://scikit-learn.org/stable/

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from globals import BOT_CONFIG


def communication_skills():
    '''Communication skills'''
    # Dataset
    corpus = []  # all samples list
    y = []  # classes
    for intent, intent_data in BOT_CONFIG['intents'].items():
        for sample in intent_data['samples']:
            corpus.append(sample)
            y.append(intent)
    # Vectorisation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    # Learning
    clf = LogisticRegression().fit(X, y)
    return {'vectorizer': vectorizer, 'clf': clf}
