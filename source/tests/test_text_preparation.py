import unittest

import pandas as pd

from source.library.text_preparation import clean
from source.tests.helpers import get_test_file_path, dataframe_to_text_file
from source.library.text_cleaning_simple import tokenize


class TestTextPreperation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reddit = pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))
        cls.reddit = reddit

    def test__count_tokens(self):
        with open(get_test_file_path('text_preparation/clean__reddit.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in self.reddit['post'].apply(clean)])
