#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from ml import communication_skills
from test_data.test_data import TEST_DATA


class Test(unittest.TestCase):

    def setUp(self):
        self.vectorizer = None
        self.clf = None
        self.get_skills()

    def get_skills(self):
        '''Get communication skills'''
        skills = communication_skills()
        self.vectorizer = skills['vectorizer']
        self.clf = skills['clf']
        self.test_data = TEST_DATA

    def get_intent(self, question):
        question_vector = self.vectorizer.transform([question])
        probas = self.clf.predict_proba(question_vector)[0]
        if max(probas) > 0.4:
            intent = self.clf.predict(question_vector)[0]
            return intent

    def test_intents(self):
        for intent, questions in self.test_data.items():
            for question in questions:
                intent2 = self.get_intent(question)
                err_msg = 'Question: `{}` -> Intent: `{}`'.format(
                    question, intent2)
                self.assertEqual(intent, intent2, err_msg)


if __name__ == '__main__':
    unittest.main()
