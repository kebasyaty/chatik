#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# импортируем библиотеку для работы с таблицами
import pandas as pd
#
from sklearn.base import BaseEstimator
# импортируем sklearn - самую популярную библиотеку с машинным обучением
from sklearn.decomposition import TruncatedSVD
# импортируем алгоритм, известный как Метод главных компонент
from sklearn.feature_extraction.text import TfidfVectorizer
#
from sklearn.neighbors import BallTree
#
from sklearn.pipeline import make_pipeline


def main():
    # читаем данные из файла
    good = pd.read_csv('good.tsv', sep='\t')
    # создадим объект, который будет преобразовывать короткие в числовые векторы
    vectorizer = TfidfVectorizer()
    # "обучаем" его на всех контекстах -> запоминаем частоту каждого слова.
    vectorizer.fit(good.context_0)
    # записываем в матрицу, сколько раз каждое слово встречалось в каждом тексте.
    matrix_big = vectorizer.transform(good.context_0)
    # print(matrix_big.shape) # размер матрицы = число фраз * размер словаря
    #
    # алгоритм будет проецировать данные в 300-мерное пространство
    svd = TruncatedSVD(n_components=300)
    # коэффициенты этого преобразования выучиваются так,
    # чтобы сохранить максимум информации от исходной матрице
    svd.fit(matrix_big)
    matrix_small = svd.transform(matrix_big)
    # в результате строк (наблюдений) столько же столбцов меньше
    # print(matrix_small.shape)
    # при этом сохранилось больше 40% информации
    # print(svd.explained_variance_ratio_.sum())
    #
    ns = NeighborSampler()
    ns.fit(matrix_small, good.reply)
    pipe = make_pipeline(vectorizer, svd, ns)
    #
    print(pipe.predict(['где ты живешь?']))
    print(pipe.predict(['ты любишь читать книги?']))
    print(pipe.predict(['а что ты делаешь в библиотеке?']))


def softmax(x):
    '''Функция для создания вероятностного распределения'''
    proba = np.exp(-x)
    return proba / sum(proba)


class NeighborSampler(BaseEstimator):
    '''Класс для случайного выбора одного из ближайших соседей'''

    def __init__(self, k=5, temperature=1.0) -> None:
        super().__init__()
        self.k = k
        self.temperature = temperature

    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)

    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(
            X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(
                index, p=softmax(distance * self.temperature)))
        return self.y_[result]


if __name__ == "__main__":
    main()
