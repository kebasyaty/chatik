# -*- coding: utf-8 -*-

# Machine learning.
# https://scikit-learn.org/stable/

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from globals import BOT_CONFIG

# Dataset
# --------------------------------------------------------------------------------------------------
corpus = []  # replicas
y = []  # their classes

for intent, intent_data in BOT_CONFIG['intents'].items():
    for sample in intent_data['samples']:
        corpus.append(sample)
        y.append(intent)

# Vectorisation
# --------------------------------------------------------------------------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print('Count feature names: ', len(vectorizer.get_feature_names()))

# Learning
# --------------------------------------------------------------------------------------------------
clf = LogisticRegression().fit(X, y)
intent = clf.predict(vectorizer.transform(['привет']))[0]
probas = clf.predict_proba(vectorizer.transform(['привет']))[0]
print('Intent: ', intent)
print('Probability:', max(probas))
print('Is match: ', bool(len(str(vectorizer.transform(['привет'])))))

# Validation
# --------------------------------------------------------------------------------------------------
scores = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LogisticRegression().fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

print('Average static match: ', sum(scores) / len(scores))


clf = LogisticRegression().fit(X, y)
print('Mean accuracy on the given test data and labels: ', clf.score(X, y))
