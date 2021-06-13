#!/usr/bin/python
# -*- coding: utf-8 -*-

# To run locally.

import telebot

from chat import Сhat

API_TOKEN = ''

bot = telebot.TeleBot(API_TOKEN)
chat = Сhat()


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, chat.get_answer(message.text.lower()))


bot.polling()
