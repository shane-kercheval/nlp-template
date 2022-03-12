import unittest

from source.executables.helpers.text_processing import tokenize, remove_stop_words
from source.tests.helpers import get_test_file_path


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
