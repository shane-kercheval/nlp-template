import unittest

from source.executables.helpers.text_processing import tokenize
from source.tests.helpers import get_test_file_path


class TestTextProcessing(unittest.TestCase):

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__timer(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = tokenize(sentence)

        with open(get_test_file_path('text_processing/tokenize__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

