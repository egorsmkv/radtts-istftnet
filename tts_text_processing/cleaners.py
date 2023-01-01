""" adapted from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
    1. "english_cleaners" for English text
    2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
         the Unidecode library (https://pypi.python.org/pypi/Unidecode)
    3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
         the symbols in symbols.py to match your data).
'''

import re
from string import punctuation
from functools import reduce
from unidecode import unidecode


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# Regular expression separating words enclosed in curly braces for cleaning
_arpa_re = re.compile(r'{[^}]+}|\S+')


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def separate_acronyms(text):
    text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
    return text


def convert_to_ascii(text):
    return unidecode(text)


def dehyphenize_compound_words(text):
    text = re.sub(r'(?<=[a-zA-Z0-9])-(?=[a-zA-Z])', ' ', text)
    return text


def remove_space_before_punctuation(text):
    return re.sub(r"\s([{}](?:\s|$))".format(punctuation), r'\1', text)


class Cleaner(object):
    def __init__(self, cleaner_names, phonemedict):
        self.cleaner_names = cleaner_names
        self.phonemedict = phonemedict

    def __call__(self, text):
        for cleaner_name in self.cleaner_names:
            sequence_fns, word_fns = self.get_cleaner_fns(cleaner_name)
            for fn in sequence_fns:
                text = fn(text)

            text = [reduce(lambda x, y: y(x), word_fns, split)
                    if split[0] != '{' else split
                    for split in _arpa_re.findall(text)]
            text = ' '.join(text)
        text = remove_space_before_punctuation(text)
        return text

    def get_cleaner_fns(self, cleaner_name):
        if cleaner_name == 'ukrainian_cleaners':
            sequence_fns = [lowercase, collapse_whitespace]
            word_fns = []
        else:
            raise Exception("{} cleaner not supported".format(cleaner_name))

        return sequence_fns, word_fns
