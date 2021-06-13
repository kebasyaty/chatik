# -*- coding: utf-8 -*-

import random
import re

from data.nonsense import NONSENSE
from globals import BOT_CONFIG
from ml import communication_skills


class Сhat:
    '''Сhat-Bot'''

    def __init__(self) -> None:
        self.vectorizer = None
        self.clf = None
        # self.pattern = re.compile(r',+|-+|\.+|\?+|!+|\n+|:+|;+')
        self.pattern = re.compile(r'\.+|\?+|!+|\n+')
        self.get_skills()

    def get_skills(self):
        '''Get communication skills'''
        skills = communication_skills()
        self.vectorizer = skills['vectorizer']
        self.clf = skills['clf']

    @staticmethod
    def get_answer_from_nonsense(question):
        '''Get an answer from an absurd question'''
        phrases = NONSENSE.get(question.strip().lower())

        if phrases:
            return random.choice(phrases)

    def get_intents(self, question):
        if not len(question):
            return
        question_list = self.pattern.split(question)
        question_list = list(filter(lambda text: len(text), question_list))
        intents = []

        for question in question_list:
            question_vector = self.vectorizer.transform([question])
            probas = self.clf.predict_proba(question_vector)[0]
            if max(probas) > 0.4:
                intent = self.clf.predict(question_vector)[0]
                if intent != 'pre_intents':
                    intents.append(intent)

        return intents

    @staticmethod
    def get_answer_by_intents(intents):
        '''Get an answer by intent'''
        answer = []
        for intent in intents:
            if intent in BOT_CONFIG['intents']:
                phrases = BOT_CONFIG['intents'][intent]['responses']
                answer.append(random.choice(phrases))
        answer = list(filter(lambda text: len(text), answer))
        return '\n'.join(answer)

    @staticmethod
    def get_failure_phrase():
        '''Get phrase stub'''
        phrases = BOT_CONFIG['failure_phrases']
        return random.choice(phrases)

    def get_answer(self, question):
        '''Get an answer to a user question'''
        # Get an answer from an absurd question
        answer = self.get_answer_from_nonsense(question)
        if answer:
            return answer

        # NLU
        intents = self.get_intents(question)

        # Get answer

        # Finding a prepared answer
        if intents:
            answer = self.get_answer_by_intents(intents)
            if answer:
                return answer

        # Use stub
        answer = self.get_failure_phrase()
        return answer
