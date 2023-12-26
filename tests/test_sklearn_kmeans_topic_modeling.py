import unittest
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from source.library.sklearn_topic_modeling import KMeansTopicExplorer
from tests.helpers import get_test_file_path, dataframe_to_text_file


class TestSklearnTopicModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paragraphs = pd.read_parquet(get_test_file_path('datasets/un_debates_paragraphs__sample.parquet'))  # noqa
        cls.paragraphs = paragraphs
        num_topics = 10
        cls.num_topics = num_topics

        stop_words = STOP_WORDS
        stop_words |= {'ll', 've'}

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        k_means_model = KMeans(n_clusters=10, n_init='auto', random_state=42)
        k_means_model.fit(tfidf_vectors)
        cls.k_means_explorer = KMeansTopicExplorer(
            model=k_means_model,
            vectorizer=tfidf_vectorizer,
            vectors=tfidf_vectors
        )

    def test__extract_topic_dictionary(self):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        top_n_tokens = 7
        topics_dict = self.k_means_explorer.extract_topic_dictionary(top_n_tokens=top_n_tokens)
        self.assertEqual(len(topics_dict), self.num_topics)
        self.assertEqual(list(topics_dict.keys()), list(range(self.num_topics)))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))
        self.assertTrue(all([len(x) == top_n_tokens for x in topics_dict.values()]))

        # check that all token/value pairs have a length of two
        self.assertTrue(all(flatten([
            [len(token_tuple) == 2 for token_tuple in token_list]
            for token_list in topics_dict.values()
        ])))
        # check that all token/value pairs have a string as the first instance
        self.assertTrue(all(flatten([
            [isinstance(token_tuple[0], str) for token_tuple in token_list]
            for token_list in topics_dict.values()
        ])))
        # check that all token/value pairs have a float as the second instance
        self.assertTrue(all(flatten([
            [isinstance(token_tuple[1], float) for token_tuple in token_list]
            for token_list in topics_dict.values()
        ])))

    def test__extract_topic_labels(self):
        topics_dict = self.k_means_explorer.extract_topic_dictionary()
        labels = self.k_means_explorer.extract_topic_labels(
            token_separator='|',
            num_tokens_in_label=3
        )
        self.assertEqual(topics_dict.keys(), labels.keys())
        self.assertTrue(all([isinstance(x, str) for x in labels.values()]))
        self.assertTrue(all([len(x.split('|')) == 3 for x in labels.values()]))

    def test__extract_topic_dataframe(self):
        topics_dict = self.k_means_explorer.extract_topic_dictionary()
        list(topics_dict.keys())

        top_n_tokens = 7
        num_tokens_in_label = 3
        topics_df = self.k_means_explorer.extract_topic_dataframe(
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
        )
        self.assertEqual(len(topics_df), self.num_topics * top_n_tokens)
        self.assertEqual(topics_df['topic'].unique().tolist(), list(topics_dict.keys()))
        self.assertEqual(topics_df['token_index'].unique().tolist(), list(range(0, top_n_tokens)))

        dataframe_to_text_file(
            topics_df,
            get_test_file_path('topic_modeling/k_means__extract_topic_dataframe.txt')
        )

    def test__calculate_topic_sizes(self):
        # first test with low sample sizes; we need to make sure the correct number of topics are
        # returned
        examples = self.paragraphs['text'].iloc[0:2]
        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=examples,
            relative_sizes=False
        )
        self.assertEqual(len(sizes), 10)
        self.assertEqual(len(examples), sizes.sum())

        examples = self.paragraphs['text'].iloc[0:2]
        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=examples,
            relative_sizes=True
        )
        self.assertEqual(len(sizes), 10)
        self.assertEqual(sizes.sum(), 1)

        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=None,
            relative_sizes=False
        )
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(sizes.sum(), 2000)

        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=None,
            relative_sizes=True
        )
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(round(sizes.sum(), 8), 1)

        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=self.paragraphs['text'].sample(100, random_state=42),
            relative_sizes=False
        )
        self.assertEqual(sizes.sum(), 100)

        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=self.paragraphs['text'].sample(100, random_state=42),
            relative_sizes=True
        )
        self.assertEqual(round(sizes.sum(), 8), 1)

        sizes = self.k_means_explorer.calculate_topic_sizes(
            text_series=self.paragraphs['text'],
            relative_sizes=False
        )
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(sizes.sum(), len(self.paragraphs))

    def test__plot_topic_sizes(self):
        fig = self.k_means_explorer.plot_topic_sizes(text_series=self.paragraphs['text'])
        self.assertIsNotNone(fig)

    def test__plot_topics(self):
        fig = self.k_means_explorer.plot_topics(
            top_n_tokens=5,
            num_tokens_in_label=2,
        )
        self.assertIsNotNone(fig)

    def test__extract_top_examples(self):
        examples = self.k_means_explorer.extract_top_examples(
            text_series=self.paragraphs['text'],
            top_n_examples=7,
            max_num_characters=100,
            surround_matches="*",
            num_tokens_in_label=2
        )
        self.assertEqual(len(examples), 7 * self.num_topics)
        dataframe_to_text_file(
            examples,
            get_test_file_path('topic_modeling/k_means__extract_top_examples__default.txt')
        )

        examples = self.k_means_explorer.extract_top_examples(
            text_series=self.paragraphs['text'],
            top_n_examples=7,
            max_num_characters=100,
            surround_matches=None,
            num_tokens_in_label=2
        )
        for index in examples.index:
            self.assertEqual(
                examples.loc[index]['text'],
                self.paragraphs['text'].loc[index][0:100]
            )

    def test__get_topic_sizes_per_segment(self):
        topic_sizes_per_year = self.k_means_explorer.get_topic_sizes_per_segment(
            df=self.paragraphs,
            text_column='text',
            segment_column='year',
            token_separator='**',
            num_tokens_in_label=2,
        )
        self.assertEqual(
            len(topic_sizes_per_year),
            len(self.paragraphs['year'].unique()) * self.num_topics
        )

        self.assertTrue(
            (round(topic_sizes_per_year.groupby('year')['relative_size'].sum(), 5) == 1).all()
        )
        dataframe_to_text_file(
            topic_sizes_per_year,
            get_test_file_path('topic_modeling/k_means__get_topic_sizes_per_segment__year.txt')
        )


if __name__ == '__main__':
    unittest.main()
