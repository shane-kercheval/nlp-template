import numpy
import pandas as pd


def extract_topic_dictionary(model, features: numpy.ndarray, top_n_tokens: int = 10):
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

    Args:
         model:
            the topic model (e.g. sci-kit learn NMF, LatentDirichletAllocation)
         features:
            a numpy array containing the the features (e.g. tokens/bi-grams that correspond to the fitted model
        top_n_tokens:
            the number of tokens to extract from the model (which have the highest scores per topic)
    """
    topics = dict()
    for topic, tokens in enumerate(model.components_):
        total = tokens.sum()
        largest = tokens.argsort()[::-1] # invert sort order
        topics[topic + 1] = [(features[largest[i]], abs(tokens[largest[i]]*100.0/total)) for i in range(0, top_n_tokens)]
    return topics


def calculate_topic_sizes():
    pass
    # w_matrix_unigrams.sum(axis=0) / w_matrix_unigrams.sum() * 100.0
    #
    # W_text_matrix.sum(axis=0)/W_text_matrix.sum()*100.0
    # W_lda_para_matrix.sum(axis=0)/W_lda_para_matrix.sum()*100.0
    #



def topic_dict_to_labels(topic_dictionary: dict, num_tokens_in_name: int=3):
    return {topic:' | '.join([y[0] for y in x[0:num_tokens_in_name]]) for topic, x in topic_dictionary.items()}

# name_lookup = topic_dictionary_to_names(topic_dictionary, num_tokens_in_name=2)
# name_lookup


def topics_to_dataframe(model, features: list, top_n_tokens: int = 10,
                        num_tokens_in_name: int = 2) -> pd.DataFrame:
    topic_dictionary = topics_to_dictionary(model, features, top_n_tokens)
    name_lookup = topic_dictionary_to_names(topic_dictionary, num_tokens_in_name=num_tokens_in_name)

    topic_tokens = pd.DataFrame(topic_dictionary)
    topics = topic_tokens.columns
    topic_tokens = topic_tokens.reset_index().rename(columns={'index': 'token'})
    topic_tokens = pd.melt(topic_tokens, id_vars='token', value_vars=list(topics), var_name='topic')
    topic_tokens = topic_tokens.assign(
        **pd.DataFrame(topic_tokens['value'].tolist(), columns=['tokens', 'value']))
    topic_tokens['label'] = topic_tokens['topic'].apply(lambda x: name_lookup[x])
    return topic_tokens


# topic_df = topics_to_dataframe(
#     model=nmf_unigrams,
#     features=tfidf_vectorizer_unigrams.get_feature_names_out(),
#     top_n_tokens=10,
#     num_tokens_in_name=2,
# )
# topic_df

def plot_topics(topics_df):
    import plotly_express as px

    fig = px.bar(
        topics_df,
        x='value',
        y='tokens',
        facet_col='label',
        facet_col_wrap=3,
        facet_col_spacing=0.2,
        labels={
            'tokens': '',
            'label': '',
        },
        width=900,
        height=1000,
        title="Topics in NMF model (Unigrams)"
    )
    fig.update_yaxes(matches=None, showticklabels=True, autorange="reversed")
    # fig.update_xaxes(matches=None)

    return fig