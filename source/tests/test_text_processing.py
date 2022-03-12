import unittest

import pandas as pd

from source.executables.helpers.text_processing import tokenize, remove_stop_words, prepare, count_tokens
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


class TestTextProcessing(unittest.TestCase):

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__tokenize(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = tokenize(sentence)

        with open(get_test_file_path('text_processing/tokenize__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

    def test__remove_stop_words(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = tokenize(sentence)
        tokens = remove_stop_words(tokens)
        with open(get_test_file_path('text_processing/remove_stop_words__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = tokenize(sentence)
        tokens = remove_stop_words(tokens, include_stop_words=['sentence'], exclude_stop_words=['a'])
        with open(get_test_file_path('text_processing/remove_stop_words__include_exclude.txt'), 'w') as file:
            file.write('|'.join(tokens))

    def test__prepare(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = prepare(sentence)
        with open(get_test_file_path('text_processing/prepare__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

    def test__count_tokens(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. Nevermind, it doesn't have numbers. It's a dumb sentence."  # noqa
        tokens = prepare(sentence)

        dataframe_to_text_file(count_tokens(tokens, min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__list__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(tokens, min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__list__min_freq_2.txt'))

        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__list_of_lists__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__list_of_lists__min_freq_2.txt'))

        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__series_strings__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__series_strings__min_freq_2.txt'))

        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__series_list__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__series_list__min_freq_2.txt'))
