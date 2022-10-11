## Команды для запуска различных вариантов тестирования
### issue-01 `doctest`
```python
python -m doctest -o NORMALIZE_WHITESPACE -v issue-01.py > .\result_1.txt
```
### issue-02 `pytest`
```python
python -m pytest -v .\issue-02.py > .\result_2.txt
```
### issue-03 `unittest`
```python
python -m unittest -v .\issue-03.py  2> result_3.txt 
```

### issue-04 `pytest`
```python 
python -m pytest -v .\issue-04.py > result_4.txt
```

### issue-05 `unittest.mock`
1. Запустим тест с модулем для подсчета покрытия
```python
python -m coverage run -m unittest -v .\issue-05.py 2> result_5.txt
```
2. Сохраним отчет по покрытию в виде html файла 
```python
python -m coverage html 
```