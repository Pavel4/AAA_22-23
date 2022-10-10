"""Morse Code Translator"""

LETTER_TO_MORSE = {
    'A': '.-', 'B': '-...', 'C': '-.-.',
    'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..',
    'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-',
    'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..', '1': '.----',
    '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.', '0': '-----',
    ', ': '--..--', '.': '.-.-.-', '?': '..--..',
    '/': '-..-.', '-': '-....-', '(': '-.--.', ')': '-.--.-',
    ' ': ' '
}

MORSE_TO_LETTER = {
    morse: letter
    for letter, morse in LETTER_TO_MORSE.items()
}


def encode(message: str) -> str:
    """
    Кодирует строку в соответсвие с таблицей азбуки Морзе

    >>> encode('SOS')
    '... --- ...'

    >>> encode('PYTHON')
    '.--. -.-- - .... --- -.'

    >>> encode('AVITO')
    '.- ...- .. - ---'

    """
    encoded_signs = [
        LETTER_TO_MORSE[letter] for letter in message
    ]

    return ' '.join(encoded_signs)


def encode_test() -> None:
    """ Тестирующая функция для функции encode()"""
    dict_morse = {
        'SOS': '... --- ...',
        'PYTHON': '.--. -.-- - .... --- -.',
        'Python': '.--. -.-- - .... --- -.',
        'AVITO': '.- ...- .. - ---'
    }

    for word, morse_key in dict_morse.items():
        try:
            assert encode(word) == morse_key
        except KeyError:
            print(f'Not all characters of a word \'{word}\' are contained in Morse code dictionary.')
        except AssertionError:
            print(f'Failed!\nExpected: {morse_key}\nGot: {encode(word)}')


if __name__ == '__main__':
    encode_test()
