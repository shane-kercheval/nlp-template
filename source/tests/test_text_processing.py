import os
import unittest

import pandas as pd

from source.executables.helpers.text_processing import tokenize, remove_stop_words, prepare, count_tokens, term_frequency, inverse_document_frequency
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


class TestTextProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        un_debates = pd.read_csv(get_test_file_path('text_processing/un-general-debates-blueprint__sample.csv'))
        un_debates['tokens'] = un_debates['text'].map(prepare)
        cls.un_debates = un_debates

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__tokenize(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."
        tokens = tokenize(sentence)

        with open(get_test_file_path('text_processing/tokenize__simple.txt'), 'w') as file:
            file.write('|'.join(tokens))

        token_list = self.un_debates.text.map(tokenize)
        token_list = ['|'.join(x) for x in token_list]
        with open(get_test_file_path('text_processing/tokenize__un_debates.txt'), 'w') as file:
            file.write('\n'.join(token_list))

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

        token_list = ['|'.join(x) for x in self.un_debates['tokens']]
        with open(get_test_file_path('text_processing/prepare__un_debates.txt'), 'w') as file:
            file.write('\n'.join(token_list))

    def test__count_tokens(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. Nevermind, it doesn't have numbers. It's a dumb sentence."  # noqa
        tokens = prepare(sentence)

        dataframe_to_text_file(count_tokens(tokens, min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__list__min_freq_1.txt'))

        self.assertTrue((count_tokens(tokens, min_frequency=1, count_once_per_doc=True)['frequency'] == 1).all())

        dataframe_to_text_file(count_tokens(tokens, min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__list__min_freq_2.txt'))

        self.assertEqual(count_tokens(tokens, min_frequency=2, count_once_per_doc=True).shape, (0, 1))

        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__list_of_lists__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__list_of_lists__min_freq_2.txt'))

        self.assertTrue((count_tokens([tokens, tokens], min_frequency=1, count_once_per_doc=True)['frequency'] == 2).all())

        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__series_strings__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__series_strings__min_freq_2.txt'))

        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=1),
                               get_test_file_path('text_processing/count_tokens__series_list__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__series_list__min_freq_2.txt'))

        counts = count_tokens(pd.Series([tokens, tokens]), min_frequency=1, count_once_per_doc=True)
        self.assertTrue((counts['frequency'] == 2).all())
        self.assertFalse(counts.index.duplicated().any())

        dataframe_to_text_file(count_tokens(self.un_debates['tokens'], min_frequency=2),
                               get_test_file_path('text_processing/count_tokens__un_debates.txt'))

        example = [['a', 'b', 'b'], ['b', 'c', 'c']]
        self.assertEqual(count_tokens(example, min_frequency=1).to_dict(), {'frequency': {'b': 3, 'c': 2, 'a': 1}})
        self.assertEqual(count_tokens(example, min_frequency=2).to_dict(), {'frequency': {'b': 3, 'c': 2}})
        self.assertEqual(count_tokens(example, min_frequency=1, count_once_per_doc=True).to_dict(),
                         {'frequency': {'b': 2, 'a': 1, 'c': 1}})
        self.assertEqual(count_tokens(example, min_frequency=2, count_once_per_doc=True).to_dict(),
                         {'frequency': {'b': 2}})

    def test__term_frequency(self):
        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', min_frequency=3)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 3).all())
        self.assertEqual(term_freq.index[0], 'nations')
        self.assertEqual(term_freq.index[1], 'united')
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_processing/term_frequency__un_debates__min_freq_3.txt'))

        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', segment_columns='year', min_frequency=3)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 3).all())
        self.assertEqual(term_freq.index[0], (1970, 'united'))
        self.assertEqual(term_freq.index[1], (1970, 'nations'))
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_processing/term_frequency__un_debates__by_year__min_freq_3.txt'))

        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', segment_columns=['year', 'country'], min_frequency=20)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 20).all())
        self.assertEqual(term_freq.index[0], (1970, 'AUS', 'nations'))
        self.assertEqual(term_freq.index[1], (1970, 'AUS', 'united'))
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_processing/term_frequency__un_debates__by_year_country__min_freq_3.txt'))

    def test__inverse_document_frequency(self):
        idf = inverse_document_frequency(self.un_debates['tokens'])
        self.assertFalse(idf.index.duplicated().any())
        counts = count_tokens(self.un_debates['tokens'], min_frequency=2, count_once_per_doc=True)
        self.assertEqual(set(idf.index), set(counts.index))

        dataframe_to_text_file(idf,
                               get_test_file_path('text_processing/inverse_document_frequency__un_debates__min_freq_2.txt'))

