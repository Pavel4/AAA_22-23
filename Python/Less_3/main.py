import json
import keyword


class JSONToObject:

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


class Advert(JSONToObject):
    def __init__(self, mapping: [dict, str]) -> None:
        JSONToObject.__init__(self, mapping)
        if not hasattr(self, 'price'):
            self._price = 0

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value: [int, float]):
        if value < 0:
            raise ValueError('must be >= 0')
        self._price = value


if __name__ == "__main__":
    json_1 = {
        "title": "iPhone X",
        "price": 100,
        "location": {
            "address": "город Самара, улица Мориса Тореза, 50",
            "metro_stations": ["Спортивная", "Гагаринская"]
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

    lesson = json.loads(lesson_str)

    lesson_ad = Advert(json_1)

    print(lesson_ad.price)
