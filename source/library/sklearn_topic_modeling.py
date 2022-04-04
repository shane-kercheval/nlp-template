from typing import Union

import numpy
import pandas as pd
import plotly_express as px
from plotly.graph_objs import _figure  # noqa


class TopicModelExplorer:
    def __init__(self, model, vectorizer: numpy.ndarray):
        """
        Args:
            model:
                the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
            vectorizer:
                TBD
        """
        self._model = model
        self._vectorizer = vectorizer
        self._token_names = vectorizer.get_feature_names_out()

    @property
    def token_names(self):
        return self._token_names

    def extract_topic_dictionary(self, top_n_tokens: int = 10) -> dict:
        """
        This function extracts a dictionary from the topic model in the following format:

            {
                1:
                    [
                        ('token 1', 29.185453377735968),
                        ('token 2', 1.7306266040805285),
                        ('token 3', 1.3485401662261807),
                        ...
                    ],
                ...
            }

        Where the keys correspond to the topic indexes, starting with 1 (i.e. the first topic), and the values
        contain a list of tuples corresponding to the top tokens (e.g. uni-grams/bi-grams) and contribution
        values for that topic.

        This code is modified from:
            https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch08/Topic_Modeling_Clustering.ipynb
        Args:
            top_n_tokens:
                the number of tokens to extract from the model (which have the highest scores per topic)
        """
        topics = dict()
        for topic, tokens in enumerate(self._model.components_):
            total = tokens.sum()
            largest = tokens.argsort()[::-1]  # invert sort order
            topics[topic + 1] = [(self._token_names[largest[i]], abs(tokens[largest[i]] * 100.0 / total))
                                 for i in range(0, top_n_tokens)]
        return topics

    def extract_topic_labels(self,
                             token_separator: str = ' | ',
                             num_tokens_in_label: int = 2) -> dict:
        """
        This function creates topic labels from a model.

        For example, if `num_tokens_in_label` is set to `2`, then it will take the top 2 words from each topic
        and return a dictionary containing the same keys as the topic_dictionary, each topic containing a
        string value such as "word-1 | word-2" which can be used as a label for graphs and tables.

        Args:
            token_separator:
                the separator string for joining the top tokens
            num_tokens_in_label:
                the number of (top) tokens to use in the label
        """
        topic_dictionary = self.extract_topic_dictionary(top_n_tokens=num_tokens_in_label)
        return {topic: token_separator.join([y[0] for y in x[0:num_tokens_in_label]])
                for topic, x in topic_dictionary.items()}

    def extract_topic_dataframe(self,
                                top_n_tokens: int = 10,
                                num_tokens_in_label: int = 2,
                                token_separator: str = ' | ') -> pd.DataFrame:
        """
        This function returns a pd.DataFrame where each row represents a single
        Args:
            top_n_tokens:
                the number of tokens to extract from the model (which have the highest scores per topic)
            num_tokens_in_label:
                the number of (top) tokens to use in the label
            token_separator:
                the separator string for joining the top tokens in the topic label
        """
        topic_labels = self.extract_topic_labels(
            token_separator=token_separator,
            num_tokens_in_label=num_tokens_in_label
        )
        # this creates a dataframe where each column corresponds to a topic, and the rows correspond to the
        # top_n_tokens
        topic_dictionary = self.extract_topic_dictionary(top_n_tokens=top_n_tokens)
        topic_tokens = pd.DataFrame(topic_dictionary)
        # add a column to indicate the index of the token (e.g. index 0 corresponds to the token with the
        # highest contribution value
        topics = topic_tokens.columns
        topic_tokens = topic_tokens.reset_index().rename(columns={'index': 'token_index'})
        # have a top_n_tokens (height) by #-topics (width) dataframe.
        # We are going to transform the dataframe from wide to long, meaning each row is going to have a
        # single token corresponding to a single topic
        topic_tokens = pd.melt(topic_tokens, id_vars='token_index', value_vars=list(topics), var_name='topic')
        # the `value` column is a word/value tuple; we need to split the tuple into separate columns
        topic_tokens = topic_tokens.assign(**pd.DataFrame(topic_tokens['value'].tolist(),
                                                          columns=['token', 'value']))
        # add the topic label for each corresponding topic for plots/tables.
        topic_tokens['topic_label'] = topic_tokens['topic'].apply(lambda x: topic_labels[x])
        topic_tokens = topic_tokens[['topic', 'token_index', 'token', 'value', 'topic_label']]
        return topic_tokens

    def calculate_topic_sizes(self, text_series: pd.Series) -> numpy.array:
        """
        Given a model and a dataset (e.g. output of fit_transform from CountVectorizer or TfidfVectorizer),
        this function calculates the relative size of the topics and returns a float/percentage value.

        Args:
            text_series:
                dataset series to transform/predict the topics on; e.g. scipy.sparse._csr.csr_matrix
        """
        vectors = self._vectorizer.transform(text_series)
        topic_predictions = self._model.transform(X=vectors)
        # column i.e. topic totals
        topic_totals = topic_predictions.sum(axis=0)
        return topic_totals / topic_predictions.sum()

    def plot_topic_sizes(self,
                         text_series: pd.Series,
                         token_separator: str = ' | ',
                         num_tokens_in_label: int = 2) -> _figure.Figure:
        """
        Given a model and a dataset (e.g. output of fit_transform from CountVectorizer or TfidfVectorizer),
        this function plots the relative size of the topics.

        Args:
            text_series:
                dataset series to transform/predict the topics on; e.g. scipy.sparse._csr.csr_matrix
            token_separator:
                the separator string for joining the top tokens in the topic label
            num_tokens_in_label:
                the number of (top) tokens to use in the label
        """
        topic_sizes = self.calculate_topic_sizes(text_series=text_series)
        topic_labels = list(self.extract_topic_labels(token_separator=token_separator,
                                                      num_tokens_in_label=num_tokens_in_label).values())
        df = pd.DataFrame({
            'Topics': topic_labels,
            'Topic Size as a Percent of the Dataset': topic_sizes,
        })

        fig = px.bar(
            df,
            x='Topic Size as a Percent of the Dataset',
            y='Topics',
            title='Size of Topics<br><sup>More than 1 topic can be assigned to a single document; '
                  'therefore, relative (percentage) sizes are provided.</sup>'
        )
        fig.update_layout(xaxis_tickformat='p')
        fig.update_yaxes(autorange="reversed")
        return fig

    def plot_topics(self,
                    top_n_tokens: int = 10,
                    num_tokens_in_label: int = 2,
                    token_separator: str = ' | ',
                    facet_col_wrap: int = 3,
                    facet_col_spacing: float = 0.2,
                    width: int = 900,
                    height: int = 900,
                    title: Union[str, None] = None) -> _figure.Figure:
        """
        This function plots the topics and top tokens.

        Args:
            top_n_tokens:
                the number of tokens to extract from the model (which have the highest scores per topic)
            num_tokens_in_label:
                the number of (top) tokens to use in the label
            token_separator:
                the separator string for joining the top tokens in the topic label
            facet_col_wrap:
                the number of columns to display regarding faceting
            facet_col_spacing:
                the spacing between the columns (0-1)
            width:
                the width of the graph
            height:
                the height of the graph
            title:
                the title of the graph, if None, a default title is provided
        """
        topics_df = self.extract_topic_dataframe(
            top_n_tokens=top_n_tokens,
            num_tokens_in_label=num_tokens_in_label,
            token_separator=token_separator
        )
        if title is None:
            title = f"Topics<br><sup>The top {top_n_tokens} tokens are displayed per topic; " \
                    f"the top {num_tokens_in_label} tokens are displayed as the topic name</sup>"
        fig = px.bar(
            topics_df,
            x='value',
            y='token',
            facet_col='topic_label',
            facet_col_wrap=facet_col_wrap,
            facet_col_spacing=facet_col_spacing,
            labels={
                'token': '',
                'topic_label': '',
            },
            width=width,
            height=height,
            title=title,
        )
        fig.update_yaxes(matches=None, showticklabels=True, autorange="reversed")
        return fig

    def extract_top_examples(self,
                             text_series: pd.Series,
                             top_n_examples: int = 5,
                             max_num_characters: int = 300,
                             surround_matches: Union[str, None] = '|',
                             num_tokens_in_label: int = 2) -> pd.DataFrame:
        """
        Extracts the top n examples for each topic (i.e. the highest matching documents).

        Returns the results as a pd.DataFrame.

        Args
            text_series:
                dataset series to transform/predict the topics on; e.g. scipy.sparse._csr.csr_matrix
            top_n_examples:
                the number of examples/documents to extract
            max_num_characters:
                the maximum number of characters to extract from the examples
            surround_matches:
                if `surround_matches` contains a string, any words within the example that matches the
                top 5 words of the corresponding topic, will be surrounded with these characters;
                if None, no replacement will be done
            num_tokens_in_label:
                 the number of (top) tokens to use in the label
        """
        new_data = self._vectorizer.transform(text_series)
        topic_predictions = self._model.transform(X=new_data)
        import regex

        top_words_per_topic = self.extract_topic_labels(
            token_separator=' | ',
            num_tokens_in_label=5,
        )
        top_words_per_topic = {topic: label.split(' | ') for topic, label in top_words_per_topic.items()}

        topic_labels = self.extract_topic_labels(
            token_separator=' | ',
            num_tokens_in_label=num_tokens_in_label,
        )

        examples = []
        for topic in range(0, topic_predictions.shape[1]):
            # print(topic)
            topic_predictions_per_doc = topic_predictions[:, topic]
            largest_indexes = (topic_predictions_per_doc * -1).argsort()
            top_x_indexes = largest_indexes[0:top_n_examples]
            # print(top_x_indexes)
            # print([x[0:100] + "||||" for x in paragraphs['text'].iloc[top_x_indexes]])
            for index in top_x_indexes:
                text = text_series.iloc[index][0:max_num_characters]

                if surround_matches is not None and surround_matches != '':
                    for word in top_words_per_topic[topic + 1]:
                        text = regex.sub(word, f'{surround_matches}{word}{surround_matches}',
                                         text,
                                         flags=regex.IGNORECASE)

                examples += [{
                    'topic index': topic,
                    'topic labels': topic_labels[topic + 1],
                    'text': text,
                    'index': index,
                }]

        return pd.DataFrame(examples)

    def get_topic_sizes_per_segment(self, df: pd.DataFrame,
                                    text_column: str,
                                    segment_column: str,
                                    token_separator: str = ' | ',
                                    num_tokens_in_label: int = 2
                                    ) -> pd.DataFrame:
        topic_labels = self.extract_topic_labels(
            token_separator=token_separator,
            num_tokens_in_label=num_tokens_in_label,
        )
        segments = df[segment_column].unique()
        segments.sort()

        def get_segment_sizes(text_series):
            sizes = self.calculate_topic_sizes(text_series=text_series)
            return sizes

        sizes_per_segment = {segment: get_segment_sizes(df[df[segment_column] == segment][text_column])
                             for segment in segments}
        segment_dict = {segment: {topic: value
                                  for topic, value in zip(topic_labels.values(), sizes_per_segment[segment])}
                        for segment in segments}
        df = pd.DataFrame(segment_dict).reset_index().rename(columns={'index': 'topic_labels'})
        column_values = df.columns
        df = pd.melt(
            df,
            id_vars='topic_labels',
            value_vars=list(column_values),
            var_name=segment_column,
            value_name='relative_size'
        )
        return df
