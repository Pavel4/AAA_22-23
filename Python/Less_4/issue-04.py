import sys
import numpy as np
from typing import List, Tuple


def fit_transform(*args: str) -> List[Tuple[str, List[int]]]:
    """
    fit_transform(iterable)
    fit_transform(arg1, arg2, *args)
    """
    if len(args) == 0:
        raise TypeError('expected at least 1 arguments, got 0')

    categories = args if isinstance(args[0], str) else list(args[0])
    uniq_categories = set(categories)
    bin_format = f'{{0:0{len(uniq_categories)}b}}'

    seen_categories = dict()
    transformed_rows = []

    for cat in categories:
        bin_view_cat = (int(b) for b in bin_format.format(1 << len(seen_categories)))
        seen_categories.setdefault(cat, list(bin_view_cat))
        transformed_rows.append((cat, seen_categories[cat]))

    return transformed_rows


cities = ['Moscow', 'New York', 'Moscow', 'London']
exp_transformed_cities = [
    ('Moscow', [0, 0, 1]),
    ('New York', [0, 1, 0]),
    ('Moscow', [0, 0, 1]),
    ('London', [1, 0, 0]),
]


def test_equal():
    try:
        assert exp_transformed_cities == fit_transform(*cities)
    except AssertionError:
        print('Expected and received data are not equal!')


def test_in():
    new_city = ('Fryazino', [1, 0, 0])
    try:
        assert new_city in fit_transform(*cities)
    except AssertionError:
        print(f'City {new_city} not found!', file=sys.stderr)


def test_input_type():
    new_city = ['Nizhny', 'Novgorod']
    try:
        assert isinstance(new_city, str)
    except AssertionError:
        print('Got value is not a string!', file=sys.stderr)


def test_collision():
    transformed_cities = dict(fit_transform(*cities))
    n_obj = len(transformed_cities)
    zero_array = np.zeros(n_obj)
    one_array = np.ones(n_obj)
    for value in transformed_cities.values():
        zero_array = np.array(zero_array + value)
    try:
        assert (zero_array == one_array).all()
    except AssertionError:
        print('Found collision!', file=sys.stderr)
