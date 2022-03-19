import unittest

import pandas as pd

from source.executables.helpers.text_cleaning_simple import prepare
from source.executables.helpers.text_analysis import count_tokens, term_frequency, \
    inverse_document_frequency, tf_idf, get_context_from_keyword, count_keywords, count_keywords_by, \
    count_text_patterns
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


class TestTextProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        un_debates = pd.read_csv(get_test_file_path('text_analysis/un-general-debates-blueprint__sample.csv'))
        un_debates['tokens'] = un_debates['text'].map(prepare)
        cls.un_debates = un_debates

        cls.dumb_sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. It's a dumb sentence."

    def test__count_tokens(self):
        sentence = "This is a sentence; it has punctuation, etc.. It also has numbers. Nevermind, it doesn't have numbers. It's a dumb sentence."  # noqa
        tokens = prepare(sentence)

        dataframe_to_text_file(count_tokens(tokens, min_frequency=1),
                               get_test_file_path('text_analysis/count_tokens__list__min_freq_1.txt'))
        self.assertTrue((count_tokens(tokens, min_frequency=1, count_once_per_doc=True)['frequency'] == 1).all())

        dataframe_to_text_file(count_tokens(tokens, min_frequency=2),
                               get_test_file_path('text_analysis/count_tokens__list__min_freq_2.txt'))

        self.assertEqual(count_tokens(tokens, min_frequency=2, count_once_per_doc=True).shape, (0, 1))

        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=1),
                               get_test_file_path('text_analysis/count_tokens__list_of_lists__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens([tokens, tokens], min_frequency=2),
                               get_test_file_path('text_analysis/count_tokens__list_of_lists__min_freq_2.txt'))

        self.assertTrue((count_tokens([tokens, tokens], min_frequency=1, count_once_per_doc=True)['frequency'] == 2).all())

        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=1),
                               get_test_file_path('text_analysis/count_tokens__series_strings__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series(tokens), min_frequency=2),
                               get_test_file_path('text_analysis/count_tokens__series_strings__min_freq_2.txt'))

        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=1),
                               get_test_file_path('text_analysis/count_tokens__series_list__min_freq_1.txt'))
        dataframe_to_text_file(count_tokens(pd.Series([tokens, tokens]), min_frequency=2),
                               get_test_file_path('text_analysis/count_tokens__series_list__min_freq_2.txt'))

        counts = count_tokens(pd.Series([tokens, tokens]), min_frequency=1, count_once_per_doc=True)
        self.assertTrue((counts['frequency'] == 2).all())
        self.assertFalse(counts.index.duplicated().any())

        dataframe_to_text_file(count_tokens(self.un_debates['tokens'], min_frequency=2),
                               get_test_file_path('text_analysis/count_tokens__un_debates.txt'))

        example = [['a', 'b', 'b'], ['b', 'c', 'c']]
        self.assertEqual(count_tokens(example, min_frequency=1).to_dict(), {'frequency': {'b': 3, 'c': 2, 'a': 1}})
        self.assertEqual(count_tokens(example, min_frequency=2).to_dict(), {'frequency': {'b': 3, 'c': 2}})
        self.assertEqual(count_tokens(example, min_frequency=1, count_once_per_doc=True).to_dict(),
                         {'frequency': {'b': 2, 'a': 1, 'c': 1}})
        self.assertEqual(count_tokens(example, min_frequency=2, count_once_per_doc=True).to_dict(),
                         {'frequency': {'b': 2}})

    def test__count_text_patterns(self):
        result = count_text_patterns(documents=self.dumb_sentence, pattern=r"\w{5,}")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 2, 'punctuation': 1, 'numbers': 1})

        result = count_text_patterns(documents=self.dumb_sentence, pattern=r"\w{5,}", min_frequency=2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 2})

        result = count_text_patterns(documents=[self.dumb_sentence], pattern=r"\w{5,}")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 2, 'punctuation': 1, 'numbers': 1})

        result = count_text_patterns(documents=[self.dumb_sentence, self.dumb_sentence], pattern=r"\w{5,}")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 4, 'punctuation': 2, 'numbers': 2})

        result = count_text_patterns(documents=pd.Series([self.dumb_sentence, self.dumb_sentence]), pattern=r"\w{5,}", min_frequency=3)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 4})

        result = count_text_patterns(documents=pd.Series([self.dumb_sentence, '']), pattern=r"\w{5,}", min_frequency=1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict()['frequency'], {'sentence': 2, 'punctuation': 1, 'numbers': 1})

    def test__term_frequency(self):
        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', min_frequency=3)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 3).all())
        self.assertEqual(term_freq.index[0], 'nations')
        self.assertEqual(term_freq.index[1], 'united')
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_analysis/term_frequency__un_debates__min_freq_3.txt'))

        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', segment_columns='year', min_frequency=3)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 3).all())
        self.assertEqual(term_freq.index[0], (1970, 'united'))
        self.assertEqual(term_freq.index[1], (1970, 'nations'))
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_analysis/term_frequency__un_debates__by_year__min_freq_3.txt'))

        term_freq = term_frequency(df=self.un_debates, tokens_column='tokens', segment_columns=['year', 'country'], min_frequency=20)
        self.assertFalse(term_freq.index.duplicated().any())
        self.assertTrue((term_freq['frequency'] >= 20).all())
        self.assertEqual(term_freq.index[0], (1970, 'AUS', 'nations'))
        self.assertEqual(term_freq.index[1], (1970, 'AUS', 'united'))
        dataframe_to_text_file(term_freq,
                               get_test_file_path('text_analysis/term_frequency__un_debates__by_year_country__min_freq_3.txt'))

    def test__inverse_document_frequency(self):
        idf = inverse_document_frequency(self.un_debates['tokens'])
        self.assertFalse(idf.index.duplicated().any())
        counts = count_tokens(self.un_debates['tokens'], min_frequency=2, count_once_per_doc=True)
        self.assertEqual(set(idf.index), set(counts.index))

        dataframe_to_text_file(idf,
                               get_test_file_path('text_analysis/inverse_document_frequency__un_debates__min_freq_2.txt'))

    def test__tf_idf(self):
        tf_idf_df = tf_idf(df=self.un_debates, tokens_column='tokens')
        self.assertFalse(tf_idf_df.isna().any().any())
        dataframe_to_text_file(tf_idf_df,
                               get_test_file_path('text_analysis/tf_idf__un_debates__default.txt'))

        tf_idf_df = tf_idf(df=self.un_debates, tokens_column='tokens', segment_columns='year',
                           min_frequency_document=10, min_frequency_corpus=10)
        self.assertFalse(tf_idf_df.isna().any().any())
        dataframe_to_text_file(tf_idf_df,
                               get_test_file_path('text_analysis/tf_idf__un_debates__by_year.txt'))

        tf_idf_df = tf_idf(df=self.un_debates, tokens_column='tokens', segment_columns=['year', 'country'],
                           min_frequency_document=5, min_frequency_corpus=10)
        self.assertFalse(tf_idf_df.isna().any().any())
        dataframe_to_text_file(tf_idf_df,
                               get_test_file_path('text_analysis/tf_idf__un_debates__by_year_country.txt'))

    def test__get_context_from_keyword(self):
        context_list = get_context_from_keyword(
            documents=pd.Series([], dtype=object),
            keyword='sentence',
            pad_context=False,
            num_samples=10,
            window_width=35,
            keyword_wrap='||',
            random_seed=42
        )
        self.assertEqual(context_list, [])

        documents = pd.Series(self.dumb_sentence)
        context_list = get_context_from_keyword(
            documents=documents,
            keyword='sentence',
            pad_context=False,
            num_samples=100,
            window_width=35,
            keyword_wrap='||',
            random_seed=42
        )
        self.assertEqual(len(context_list), 2)
        with open(get_test_file_path('text_analysis/get_context_from_keyword__sentence.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in context_list])

        context_list = get_context_from_keyword(
            documents=self.un_debates['text'],
            keyword='During',
            pad_context=True,
            num_samples=100,
            window_width=50,
            keyword_wrap='||',
            random_seed=42
        )

        self.assertEqual(len(context_list), 100)
        with open(get_test_file_path('text_analysis/get_context_from_keyword__un_debates__during.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in context_list])

        context_list = get_context_from_keyword(
            documents=self.un_debates['text'],
            keyword='united',
            pad_context=True,
            num_samples=100,
            window_width=50,
            keyword_wrap='||',
            random_seed=42
        )

        self.assertEqual(len(context_list), 100)
        with open(get_test_file_path('text_analysis/get_context_from_keyword__un_debates__united.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in context_list])

    def test__count_keywords(self):
        tokens = ['this', 'is', 'a', 'list', 'of', 'keywords', 'to', 'count', 'all', 'of', 'the', 'keywords']
        keyword_count = count_keywords(tokens=tokens, keywords=['keywords', 'count', 'of', 'does not exist'])
        self.assertEqual(keyword_count, [2, 1, 2, 0])

        keyword_count = count_keywords(tokens=tokens,
                                       keywords=['does not exist'])
        self.assertEqual(keyword_count, [0])

    def test__count_keywords_by(self):
        keywords = ['united', 'nuclear', 'terrorism', 'climate']
        keyword_counts_per_year = count_keywords_by(df=self.un_debates,
                                                    by='year',
                                                    tokens='tokens',
                                                    keywords=keywords)

        dataframe_to_text_file(keyword_counts_per_year,
                               get_test_file_path('text_analysis/count_keywords_by__un_debates__year.txt'))

        # get the total number of documents per year; when we count once per document, we should not go over
        # this number
        num_speeches_per_year = self.un_debates.groupby('year').size()
        self.assertEqual(num_speeches_per_year.index.tolist(),
                         keyword_counts_per_year['united'].index.tolist())
        self.assertTrue((keyword_counts_per_year['united'] > num_speeches_per_year).all())  # noqa

        keyword_counts_per_year = count_keywords_by(df=self.un_debates,
                                                    by='year',
                                                    tokens='tokens',
                                                    keywords=keywords,
                                                    count_once_per_doc=True)

        self.assertEqual(num_speeches_per_year.index.tolist(),
                         keyword_counts_per_year['united'].index.tolist())
        self.assertTrue((keyword_counts_per_year['united'] <= num_speeches_per_year).all())  # noqa

        dataframe_to_text_file(keyword_counts_per_year,
                               get_test_file_path('text_analysis/count_keywords_by__un_debates__year__count_once.txt'))
