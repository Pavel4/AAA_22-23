import unittest
from unittest.mock import patch
import urllib.request
import json
import datetime

API_URL = 'http://worldclockapi.com/api/json/utc/now'

YMD_SEP = '-'
YMD_SEP_INDEX = 4
YMD_YEAR_SLICE = slice(None, YMD_SEP_INDEX)

DMY_SEP = '.'
DMY_SEP_INDEX = 5
DMY_YEAR_SLICE = slice(DMY_SEP_INDEX + 1, DMY_SEP_INDEX + 5)


def what_is_year_now() -> int:
    """
    Получает текущее время из API-worldclock и извлекает из поля 'currentDateTime' год

    Предположим, что currentDateTime может быть в двух форматах:
      * YYYY-MM-DD - 2019-03-01
      * DD.MM.YYYY - 01.03.2019
    """
    with urllib.request.urlopen(API_URL) as resp:
        resp_json = json.load(resp)

    datetime_str = resp_json['currentDateTime']
    if datetime_str[YMD_SEP_INDEX] == YMD_SEP:
        year_str = datetime_str[YMD_YEAR_SLICE]
    elif datetime_str[DMY_SEP_INDEX] == DMY_SEP:
        year_str = datetime_str[DMY_YEAR_SLICE]
    else:
        raise ValueError('Invalid format')

    return int(year_str)


class TestWhatIsYearNow(unittest.TestCase):
    def test_YMD_format_date(self):
        got_current_date = {'currentDateTime': '2022-10-10'}
        with patch("urllib.request.urlopen"), \
                patch("json.load", return_value=got_current_date):
            got_year = what_is_year_now()
        exp_current_year = datetime.datetime.now().year
        self.assertEqual(got_year, exp_current_year)

    def test_DMY_format_date(self):
        got_current_date = {'currentDateTime': '10.10.2022'}
        with patch("urllib.request.urlopen"), \
                patch("json.load", return_value=got_current_date):
            got_year = what_is_year_now()
        exp_current_year = datetime.datetime.now().year
        self.assertEqual(got_year, exp_current_year)

    def test_wrong_format_date(self):
        got_current_date = {'currentDateTime': '10-10-2022'}
        with patch("urllib.request.urlopen"), \
                patch("json.load", return_value=got_current_date):
            with self.assertRaises(ValueError):
                what_is_year_now()
