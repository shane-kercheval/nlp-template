from collections import Counter
from typing import List, Callable, Union
import nltk
import pandas as pd
import regex as re


def tokenize(text: str) -> list:
    """
    Transform `text` in a list of tokens.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        text: string of text
    """
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)


def get_stop_words(source: str = 'nltk') -> set:
    if source == 'nltk':
        stop_words = set(nltk.corpus.stopwords.words('english'))
    else:
        raise ValueError("Only 'nlkt' source supported for stopwords.")

    return stop_words


def remove_stop_words(tokens: list,
                      include_stop_words: Union[list, set] = None,
                      exclude_stop_words: Union[list, set] = None,
                      source: str = 'nltk') -> list:
    """
    Remove stop-words from a list of tokens.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens: list of strings
        include_stop_words: list/set of stop-words to include
        exclude_stop_words: list/set of stop-words to exclude
        source: source of the stop-words to use

    """
    stop_words = get_stop_words(source=source)

    if include_stop_words is not None:
        if isinstance(include_stop_words, list):
            include_stop_words = set(include_stop_words)
        stop_words |= include_stop_words

    if exclude_stop_words is not None:
        if isinstance(exclude_stop_words, list):
            exclude_stop_words = set(exclude_stop_words)
        stop_words -= exclude_stop_words

    return [t for t in tokens if t.lower() not in stop_words]


def prepare(text: str, pipeline: List[Callable] = None) -> list:
    """
    Transform `text` according to the pipeline, which is a list of functions to be called on text.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        text:
            string of text
        pipeline:
            A list of functions that define how `text` should be transformed. Executed in order.
            If `None`, then will process based on a general pipeline of

                `[str.lower, tokenize, remove_stop_words]`
    """
    if pipeline is None:
        pipeline = [str.lower, tokenize, remove_stop_words]
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens


def count_tokens(tokens: Union[pd.Series, list], min_frequency: int = 2) -> pd.DataFrame:  # noqa
    """
    Counts tokens, returns the results as a DataFrame with 'token' and 'frequency' columns.

    Modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens:
            either
                - a pandas Series (with strings or a list of strings)
                - a list of strings
                - a list of lists of strings
        min_frequency:
            The minimum times the token has to appear in order to be returned
    """
    if isinstance(tokens, list):
        tokens = pd.Series(tokens)

    # create counter and run through all data
    counter = Counter()

    def count(x):
        # if we pass a string it will count characters, which is not the intent
        if isinstance(x, str):
            x = [x]
        counter.update(x)

    _ = tokens.map(count)

    # transform counter into data frame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['frequency'])
    freq_df = freq_df.query('frequency >= @min_frequency')
    freq_df.index.name = 'token'
    
    return freq_df.sort_values('frequency', ascending=False)
