import unittest
import pandas as pd

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

from source.library.sklearn_topic_modeling import TopicModelExplorer
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


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
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(count_vectors)
        lda_explorer = TopicModelExplorer(model=lda_model, vectorizer=count_vectorizer)
        cls.lda_explorer = lda_explorer

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(2, 2),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42)
        _ = nmf_model.fit(tfidf_vectors)
        nmf_explorer = TopicModelExplorer(model=nmf_model, vectorizer=tfidf_vectorizer)
        cls.nmf_explorer = nmf_explorer

    def test__extract_topic_dictionary(self):
        def flatten(t):
            return [item for sublist in t for item in sublist]

        top_n_tokens = 7
        topics_dict = self.nmf_explorer.extract_topic_dictionary(top_n_tokens=top_n_tokens)

        self.assertEqual(len(topics_dict), self.num_topics)
        self.assertEqual(list(topics_dict.keys()), list(range(1, self.num_topics + 1)))
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
        del topics_dict

        top_n_tokens = 8
        topics_dict = self.lda_explorer.extract_topic_dictionary(top_n_tokens=top_n_tokens)

        self.assertEqual(len(topics_dict), self.num_topics)
        self.assertEqual(list(topics_dict.keys()), list(range(1, self.num_topics + 1)))
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
        topics_dict = self.nmf_explorer.extract_topic_dictionary()
        labels = self.nmf_explorer.extract_topic_labels(token_separator='|', num_tokens_in_label=3)
        self.assertEqual(topics_dict.keys(), labels.keys())
        self.assertTrue(all([isinstance(x, str) for x in labels.values()]))
        self.assertTrue(all([len(x.split('|')) == 3 for x in labels.values()]))

        labels = self.lda_explorer.extract_topic_labels(token_separator='|', num_tokens_in_label=3)
        self.assertEqual(topics_dict.keys(), labels.keys())
        self.assertTrue(all([isinstance(x, str) for x in labels.values()]))
        self.assertTrue(all([len(x.split('|')) == 3 for x in labels.values()]))

    def test__extract_topic_dataframe(self):
        topics_dict = self.nmf_explorer.extract_topic_dictionary()
        list(topics_dict.keys())

        top_n_tokens = 7
        num_tokens_in_label = 3
        topics_df = self.nmf_explorer.extract_topic_dataframe(
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
        )
        self.assertEqual(len(topics_df), self.num_topics * top_n_tokens)
        self.assertEqual(topics_df['topic'].unique().tolist(), list(topics_dict.keys()))
        self.assertEqual(topics_df['token_index'].unique().tolist(), list(range(0, top_n_tokens)))

        dataframe_to_text_file(
            topics_df,
            get_test_file_path('topic_modeling/nmf__extract_topic_dataframe.txt')
        )

        top_n_tokens = 7
        num_tokens_in_label = 3
        topics_df = self.lda_explorer.extract_topic_dataframe(
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
        )
        self.assertEqual(len(topics_df), self.num_topics * top_n_tokens)
        self.assertEqual(topics_df['topic'].unique().tolist(), list(topics_dict.keys()))
        self.assertEqual(topics_df['token_index'].unique().tolist(), list(range(0, top_n_tokens)))

        dataframe_to_text_file(
            topics_df,
            get_test_file_path('topic_modeling/lda__extract_topic_dataframe.txt')
        )

    def test__calculate_topic_sizes(self):
        sizes = self.nmf_explorer.calculate_topic_sizes(text_series=self.paragraphs['text'])
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(round(sizes.sum(), 8), 1)
        self.assertEqual(len(sizes), 10)

        sizes = self.nmf_explorer.calculate_topic_sizes(text_series=self.paragraphs['text'].iloc[0:2])
        self.assertEqual(len(sizes), 10)

        sizes = self.lda_explorer.calculate_topic_sizes(text_series=self.paragraphs['text'])
        self.assertEqual(len(sizes), self.num_topics)
        self.assertEqual(round(sizes.sum(), 8), 1)

        sizes = self.lda_explorer.calculate_topic_sizes(text_series=self.paragraphs['text'].iloc[0:2])
        self.assertEqual(len(sizes), 10)

    def test__plot_topic_sizes(self):
        fig = self.nmf_explorer.plot_topic_sizes(text_series=self.paragraphs['text'])
        self.assertIsNotNone(fig)

    def test__plot_topics(self):
        fig = self.nmf_explorer.plot_topics(
            top_n_tokens=5,
            num_tokens_in_label=2,
        )
        self.assertIsNotNone(fig)

    def test__extract_top_examples(self):
        examples = self.nmf_explorer.extract_top_examples(
            text_series=self.paragraphs['text'],
            top_n_examples=7,
            max_num_characters=100,
            surround_matches="*",
            num_tokens_in_label=2
        )
        self.assertEqual(len(examples), 7 * self.num_topics)
        dataframe_to_text_file(
            examples,
            get_test_file_path('topic_modeling/nmf__extract_top_examples__default.txt')
        )

        examples = self.nmf_explorer.extract_top_examples(
            text_series=self.paragraphs['text'],
            top_n_examples=7,
            max_num_characters=100,
            surround_matches=None,
            num_tokens_in_label=2
        )
        for index in range(0, len(examples)):
            self.assertEqual(
                examples.iloc[index]['text'],
                self.paragraphs['text'].iloc[examples.iloc[index]['index']][0:100]
            )

    def test__get_topic_sizes_per_segment(self):
        topic_sizes_per_year = self.nmf_explorer.get_topic_sizes_per_segment(
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

        self.assertTrue((round(topic_sizes_per_year.groupby('year').agg(sum)['relative_size'], 5) == 1).all())
        dataframe_to_text_file(
            topic_sizes_per_year,
            get_test_file_path('topic_modeling/nmf__get_topic_sizes_per_segment__year.txt')
        )


if __name__ == '__main__':
    unittest.main()
