import unittest

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS

from source.library.topic_modeling import extract_topic_dictionary
from source.tests.helpers import get_test_file_path


class TestTextPreparation(unittest.TestCase):

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
        cls.count_features = count_vectorizer.get_feature_names_out()

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(2, 2),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        cls.tfidf_features = tfidf_vectorizer.get_feature_names_out()

        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42)
        nmf_w_matrix = nmf_model.fit_transform(tfidf_vectors)
        #nmf_h_matrix = nmf_model.components_
        cls.nmf_model = nmf_model

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_w_matrix = lda_model.fit_transform(count_vectors)
        #lda_h_matrix = lda_model.components_
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


