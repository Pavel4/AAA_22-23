import json
import keyword


class ColorizeMixin:
    """
    Mixin to add the ability to change the text color
    """

    def __repr__(self, color_code: int):
        return f'\033[1;{color_code};40m'


class JSONToObject:
    """
    The class that transforms JSON objects in python objects
    """

    def __init__(self, mapping: [dict, str]) -> None:
        if type(mapping) == dict:
            for key, value in mapping.items():
                if type(value) != dict:
                    if keyword.iskeyword(key):
                        key = key + '_'
                    self.__setattr__(key, value)
                else:
                    self.__setattr__(key, JSONToObject(value))
        else:
            self.__init__(json.loads(mapping))


class Advert(ColorizeMixin, JSONToObject):
    """
    The class that dynamically creates class instance
    attributes from attributes JSON object
    """
    repr_color_code_green = 32

    def __init__(self, mapping: [dict, str]) -> None:
        JSONToObject.__init__(self, mapping)
        if not hasattr(self, 'title'):
            self._title = None
        if not hasattr(self, 'price'):
            self._price = 0

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value: [int, float]):
        if value < 0:
            raise ValueError('must be >= 0')
        self._price = value

    def __repr__(self):
        return super().__repr__(self.repr_color_code_green) \
               + f'{self.title} | {self.price} ₽'


if __name__ == "__main__":
    iphone_json = {
        "title": "iPhone X",
        "price": 100,
        "location": {
            "address": "город Самара, улица Мориса Тореза, 50",
            "metro_stations": ["Спортивная", "Гагаринская"]
        }
    }

    corgi_json = {
        "title": "Вельш-корги",
        "price": 1000,
        "class": "dogs",
        "location": {
            "address": "сельское поселение Ельдигинское, \
            поселок санатория Тишково, 25"
        }
    }

    lesson_str = """{
    "title": "python",
    "price": 0,
    "location": {
    "address": "город Москва, Лесная, 7",
    "metro_stations": ["Белорусская"]
    }
    }"""

    lesson_ad = json.loads(lesson_str)

    lesson = Advert(lesson_ad)
    corgi = Advert(corgi_json)
    iphone = Advert(iphone_json)

    print(lesson.location.address)
    print(iphone.location.address)
    print(corgi)
