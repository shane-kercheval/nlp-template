from typing import Union

import numpy
import pandas as pd
import plotly_express as px
from plotly.graph_objs import _figure  # noqa


def extract_topic_dictionary(model, features: numpy.ndarray, top_n_tokens: int = 10) -> dict:
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
         model:
            the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
         features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted
            model
        top_n_tokens:
            the number of tokens to extract from the model (which have the highest scores per topic)
    """
    topics = dict()
    for topic, tokens in enumerate(model.components_):
        total = tokens.sum()
        largest = tokens.argsort()[::-1]  # invert sort order
        topics[topic + 1] = [(features[largest[i]], abs(tokens[largest[i]]*100.0/total))
                             for i in range(0, top_n_tokens)]
    return topics


def create_topic_labels(model,
                        features: numpy.ndarray,
                        token_separator: str = ' | ',
                        top_n_tokens: int = 2) -> dict:
    """
    This function creates topic labels from a model.

    For example, if `top_n_tokens` is set to `2`, then it will take the top 2 words from each topic and return
    a dictionary containing the same keys as the topic_dictionary, each topic containing a string value
    such as "word-1 | word-2" which can be used as a label for graphs and tables.

    Args:
        model:
            the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
        features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted
            model
        token_separator:
            the separator string for joining the top tokens
        top_n_tokens:
            the number of (top) tokens to use in the label
    """
    topic_dictionary = extract_topic_dictionary(model=model, features=features)
    return {topic: token_separator.join([y[0] for y in x[0:top_n_tokens]])
            for topic, x in topic_dictionary.items()}


def extract_topic_dataframe(model,
                            features: numpy.array,
                            top_n_tokens: int = 10,
                            num_tokens_in_label: int = 2,
                            token_separator: str = ' | ') -> pd.DataFrame:
    """
    This function returns a pd.DataFrame where each row represents a single
    Args:
        model:
            the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
        features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted
            model
        top_n_tokens:
            the number of tokens to extract from the model (which have the highest scores per topic)
        num_tokens_in_label:
            the number of (top) tokens to use in the label
        token_separator:
            the separator string for joining the top tokens in the topic label
    """
    topic_labels = create_topic_labels(
        model=model,
        features=features,
        token_separator=token_separator,
        top_n_tokens=num_tokens_in_label
    )
    # this creates a dataframe where each column corresponds to a topic, and the rows correspond to the
    # top_n_tokens
    topic_dictionary = extract_topic_dictionary(model=model, features=features, top_n_tokens=top_n_tokens)
    topic_tokens = pd.DataFrame(topic_dictionary)
    # add a column to indicate the index of the token (e.g. index 0 corresponds to the token with the
    # highest contribution value
    topics = topic_tokens.columns
    topic_tokens = topic_tokens.reset_index().rename(columns={'index': 'token_index'})
    # have a top_n_tokens (height) by #-topics (width) dataframe.
    # We are going to transform the dataframe from wide to long, meaning each row is going to have a single
    # token corresponding to a single topic
    topic_tokens = pd.melt(topic_tokens, id_vars='token_index', value_vars=list(topics), var_name='topic')
    # the `value` column is a word/value tuple; we need to split the tuple into separate columns
    topic_tokens = topic_tokens.assign(**pd.DataFrame(topic_tokens['value'].tolist(),
                                                      columns=['token', 'value']))
    # add the topic label for each corresponding topic for plots/tables.
    topic_tokens['topic_label'] = topic_tokens['topic'].apply(lambda x: topic_labels[x])
    topic_tokens = topic_tokens[['topic', 'token_index', 'token', 'value', 'topic_label']]
    return topic_tokens


def calculate_topic_sizes(model, dataset) -> numpy.array:
    """
    Given a model and a dataset (e.g. output of fit_transform from CountVectorizer or TfidfVectorizer),
    this function calculates the relative size of the topics and returns a float/percentage value.

    Args:
        model:
            The model e.g. sci-kit-learn NMF or LatentDirichletAllocation
        dataset:
            dataset to transform/predict the topics on; e.g. scipy.sparse._csr.csr_matrix
    """
    topic_predictions = model.transform(X=dataset)
    # column i.e. topic totals
    topic_totals = topic_predictions.sum(axis=0)
    return topic_totals / topic_predictions.sum()


def plot_topic_sizes(model,
                     dataset,
                     features,
                     token_separator: str = ' | ',
                     top_n_tokens: int = 2) -> _figure.Figure:
    """
    Given a model and a dataset (e.g. output of fit_transform from CountVectorizer or TfidfVectorizer),
    this function plots the relative size of the topics.

    Args:
        model:
            The model e.g. sci-kit-learn NMF or LatentDirichletAllocation
        dataset:
            dataset to transform/predict the topics on; e.g. scipy.sparse._csr.csr_matrix
        features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted
            model; This is used to build the labels of the topics.
        token_separator:
            the separator string for joining the top tokens in the topic label
        top_n_tokens:
            the number of (top) tokens to use in the label
    """
    topic_sizes = calculate_topic_sizes(model=model, dataset=dataset)
    topic_labels = list(create_topic_labels(model=model,
                                            features=features,
                                            token_separator=token_separator,
                                            top_n_tokens=top_n_tokens).values())
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


def plot_topics(model,
                features: numpy.array,
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
        model:
            the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
        features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted
            model
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
    topics_df = extract_topic_dataframe(
        model=model,
        features=features,
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
