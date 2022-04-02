import unittest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS

from source.library.sklearn_topic_modeling import extract_topic_dictionary, create_topic_labels, \
    extract_topic_dataframe, calculate_topic_sizes, plot_topics
from source.tests.helpers import get_test_file_path


class TestSklearnTopicModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paragraphs = pd.read_pickle(get_test_file_path('datasets/un_debates_paragraphs__sample.pkl'))
        cls.paragraphs = paragraphs
        num_topics = 10
        cls.num_topics = num_topics

        stop_words = STOP_WORDS
        stop_words |= {'ll', 've'}

        count_vectorizer = CountVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(2, 2),
            min_df=5,
            max_df=0.7
        )
        count_vectors = count_vectorizer.fit_transform(paragraphs["text"])
        cls.count_vectors = count_vectors
        cls.count_features = count_vectorizer.get_feature_names_out()

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(2, 2),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        cls.tfidf_vectors = tfidf_vectors
        cls.tfidf_features = tfidf_vectorizer.get_feature_names_out()

        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42)
        _ = nmf_model.fit_transform(tfidf_vectors)
        cls.nmf_model = nmf_model

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        _ = lda_model.fit_transform(count_vectors)
        cls.lda_model = lda_model

    def test__extract_topic_dictionary(self):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        top_n_tokens = 7
        topics_dict = extract_topic_dictionary(
            model=self.nmf_model,
            features=self.count_features,
            top_n_tokens=top_n_tokens
        )

        self.assertEqual(len(topics_dict), self.num_topics)
        self.assertEqual(list(topics_dict.keys()), list(range(1, self.num_topics + 1)))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))

        # check that all token/value pairs have a length of two
        self.assertTrue(all(flatten([[len(token_tuple) == 2 for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa
        # check that all token/value pairs have a string as the first instance
        self.assertTrue(all(flatten([[isinstance(token_tuple[0], str) for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa
        # check that all token/value pairs have a float as the second instance
        self.assertTrue(all(flatten([[isinstance(token_tuple[1], float) for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa
        del topics_dict

        top_n_tokens = 8
        topics_dict = extract_topic_dictionary(
            model=self.lda_model,
            features=self.count_features,
            top_n_tokens=top_n_tokens
        )

        self.assertEqual(len(topics_dict), self.num_topics)
        self.assertEqual(list(topics_dict.keys()), list(range(1, self.num_topics + 1)))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))

        # check that all token/value pairs have a length of two
        self.assertTrue(all(flatten([[len(token_tuple) == 2 for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa
        # check that all token/value pairs have a string as the first instance
        self.assertTrue(all(flatten([[isinstance(token_tuple[0], str) for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa
        # check that all token/value pairs have a float as the second instance
        self.assertTrue(all(flatten([[isinstance(token_tuple[1], float) for token_tuple in token_list] for token_list in topics_dict.values()])))  # noqa

    def test__create_topic_labels(self):
        topics_dict = extract_topic_dictionary(
            model=self.nmf_model,
            features=self.count_features,
        )
        labels = create_topic_labels(model=self.nmf_model,
                                     features=self.count_features,
                                     token_separator='|',
                                     top_n_tokens=3)
        self.assertEqual(topics_dict.keys(), labels.keys())
        self.assertTrue(all([isinstance(x, str) for x in labels.values()]))
        self.assertTrue(all([len(x.split('|')) == 3 for x in labels.values()]))

        labels = create_topic_labels(model=self.nmf_model,
                                     features=self.count_features,
                                     token_separator='|',
                                     top_n_tokens=3)
        self.assertEqual(topics_dict.keys(), labels.keys())
        self.assertTrue(all([isinstance(x, str) for x in labels.values()]))
        self.assertTrue(all([len(x.split('|')) == 3 for x in labels.values()]))

    def test__extract_topic_dataframe(self):
        topics_dict = extract_topic_dictionary(
            model=self.nmf_model,
            features=self.count_features,
        )
        list(topics_dict.keys())

        top_n_tokens = 7
        num_tokens_in_label = 3
        topics_df = extract_topic_dataframe(
            model=self.nmf_model,
            features=self.count_features,
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
        )
        self.assertEqual(len(topics_df), self.num_topics * top_n_tokens)
        self.assertEqual(topics_df['topic'].unique().tolist(), list(topics_dict.keys()))
        self.assertEqual(topics_df['token_index'].unique().tolist(), list(range(0, top_n_tokens)))

        top_n_tokens = 7
        num_tokens_in_label = 3
        topics_df = extract_topic_dataframe(
            model=self.lda_model,
            features=self.count_features,
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
        )
        self.assertEqual(len(topics_df), self.num_topics * top_n_tokens)
        self.assertEqual(topics_df['topic'].unique().tolist(), list(topics_dict.keys()))
        self.assertEqual(topics_df['token_index'].unique().tolist(), list(range(0, top_n_tokens)))

    def test__calculate_topic_sizes(self):
        sizes = calculate_topic_sizes(model=self.nmf_model, dataset=self.count_vectors)
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(round(sizes.sum(), 8), 1)

        sizes = calculate_topic_sizes(model=self.lda_model, dataset=self.tfidf_vectors)
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(round(sizes.sum(), 8), 1)

    def test__plot_topics(self):
        topics_df = extract_topic_dataframe(
            model=self.lda_model,
            features=self.count_features,
            top_n_tokens=5,
            num_tokens_in_label=2,
        )
        fig = plot_topics(topics_df=topics_df)
        self.assertIsNotNone(fig)
