# -*- coding: utf-8 -*-

from data.phrases import (bye, hello, my_birthday, my_destiny,
                          my_favorite_color, my_name, pre_intents)
from data.phrases.weather import (rainy_weather, snowy_weather, sunny_weather,
                                  weather, windy_weather)

INTENTS = {
    # 'pre_intents': pre_intents.PRE_INTENTS,
    'hello': hello.HELLO,
    'bye': bye.BYE,
    'my_destiny': my_destiny.MY_DESTINY,
    'my_name': my_name.MY_NAME,
    'my_favorite_color': my_favorite_color.MY_FAVORITE_COLOR,
    'my_birthday': my_birthday.MY_BIRTHDAY,
    'rainy_weather': rainy_weather.RAINY_WEATHER,
    'snowy_weather': snowy_weather.SNOWY_WEATHER,
    'sunny_weather': sunny_weather.SUNNY_WEATHER,
    'windy_weather': windy_weather.WINDY_WEATHER,
    'weather': weather.WEATHER,
}
