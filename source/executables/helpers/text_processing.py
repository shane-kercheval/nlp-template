from typing import List, Callable, Union
import nltk
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
                      exclude_stop_words = None,
                      source: str = 'nltk') -> list:
    """
    Remove stop-words from `tokens` in a list of tokens.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens: string of text
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


def prepare(text: str, pipeline: List[Callable]) -> list:
    """
    Transform `text` according to the pipeline, which is a list of functions to be called on text.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        text: string of text
        pipeline: a list of functions that define how `text` should be transformed. Executed in order.
    """
    if pipeline is None:
        pipeline = [str.lower, tokenize, remove_stop]
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens