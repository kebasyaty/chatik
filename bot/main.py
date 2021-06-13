#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from chat import Сhat


def main():
    chat = Сhat()
    question = None

    while question not in ['exit', 'выход']:
        question = input()
        answer = chat.get_answer(question)
        print(answer)


if __name__ == "__main__":
    main()
