# -*- coding: utf-8 -*-

import re

from sklearn.pipeline import make_pipeline

from ml import communication_skills

vectorizer, svd, ns = communication_skills()
pipe = make_pipeline(vectorizer, svd, ns)


class Сhat:
    '''Сhat-Bot'''

    def __init__(self) -> None:
        self.pipe = pipe

    def get_answer(self, question):
        answer = self.pipe.predict([question])[0]
        answer = re.sub(r'\s(,|\.|!|\?|:|;)', r'\1', answer)
        return answer
