import unittest

import pandas as pd

from source.library.text_cleaning_simple import tokenize, remove_stop_words, prepare, \
    get_stop_words, get_n_grams
import source.library.regex_patterns as rx
from source.tests.helpers import get_test_file_path


class TestTextCleaningSimple(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        un_debates = pd.read_csv(get_test_file_path('datasets/un-general-debates-blueprint__sample.csv'))
        un_debates['tokens'] = un_debates['text'].map(prepare)
        cls.un_debates = un_debates

        cls.dumb_sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."

    def test__get_stop_words(self):
        stop_words = get_stop_words()
        self.assertIsInstance(stop_words, set)
        self.assertTrue(len(stop_words) > 1)
        self.assertTrue('am' in stop_words)
        self.assertFalse('stop' in stop_words)
        stop_words = get_stop_words(include_stop_words=['stop'], exclude_stop_words=['am'])
        self.assertIsInstance(stop_words, set)
        self.assertTrue(len(stop_words) > 1)
        self.assertFalse('am' in stop_words)
        self.assertTrue('stop' in stop_words)

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__tokenize(self):
        tokens = tokenize(self.dumb_sentence)

        with open(get_test_file_path('text_cleaning_simple/tokenize__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

        token_list = self.un_debates.text.map(tokenize)
        token_list = ['|'.join(x) for x in token_list]
        with open(get_test_file_path('text_cleaning_simple/tokenize__un_debates.txt'), 'w') as file:
            file.write('\n'.join(token_list))

        token_list = self.un_debates.text.apply(tokenize, pattern=rx.TOKENS)
        token_list = ['|'.join(x) for x in token_list]
        with open(get_test_file_path('text_cleaning_simple/tokenize__un_debates__pattern.txt'), 'w') as file:
            file.write('\n'.join(token_list))

    def test__remove_stop_words(self):
        tokens = tokenize(self.dumb_sentence)
        tokens = remove_stop_words(tokens)
        with open(get_test_file_path('text_cleaning_simple/remove_stop_words__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

        tokens = tokenize(self.dumb_sentence)
        tokens = remove_stop_words(tokens,
                                   stop_words=get_stop_words(include_stop_words=['sentence'], exclude_stop_words=['a']))
        with open(get_test_file_path('text_cleaning_simple/remove_stop_words__include_exclude.txt'), 'w') as file:
            file.write('|'.join(tokens))

    def test__prepare(self):
        tokens = prepare(self.dumb_sentence)
        with open(get_test_file_path('text_cleaning_simple/prepare__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

        token_list = ['|'.join(x) for x in self.un_debates['tokens']]
        with open(get_test_file_path('text_cleaning_simple/prepare__un_debates.txt'), 'w') as file:
            file.write('\n'.join(token_list))

    def test__get_n_grams(self):
        tokens = prepare(text=self.dumb_sentence, pipeline=[str.lower, tokenize])
        n_grams_list = get_n_grams(tokens=tokens)
        with open(get_test_file_path('text_cleaning_simple/get_n_grams__sentence_no_stopwords_removed.txt'), 'w') as handle:
            handle.writelines([str(x) + "\n" for x in n_grams_list])

        n_grams_list = get_n_grams(tokens=tokens, stop_words=get_stop_words())
        with open(get_test_file_path('text_cleaning_simple/get_n_grams__sentence_stopwords_removed.txt'), 'w') as handle:
            handle.writelines([str(x) + "\n" for x in n_grams_list])

        n_gram_series = self.un_debates['text'].\
            apply(prepare, pipeline=[str.lower, tokenize]).\
            apply(get_n_grams, n=2, stop_words=get_stop_words())

        self.assertIsInstance(n_gram_series, pd.Series)
        self.assertEqual(len(n_gram_series), len(self.un_debates['text']))

        with open(get_test_file_path('text_cleaning_simple/get_n_grams__un_debates.txt'), 'w') as handle:
            handle.writelines([str(x) + "\n" for x in n_gram_series])
