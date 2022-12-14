from typing import List, Callable, Union, Set

import nltk
import regex as re

import source.library.regex_patterns as rx


def tokenize(text: str, pattern: str = rx.TOKENS_SIMPLE) -> List[str]:
    """
    Transform `text` in a list of tokens.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        text: string of text
        pattern: string with regex pattern for extracting tokens
    """
    return re.findall(pattern, text, re.VERBOSE)


def get_stop_words(source: str = 'nltk',
                   include_stop_words: Union[Set[str], List[str]] = None,
                   exclude_stop_words: Union[Set[str], List[str]] = None) -> Set[str]:
    """
    Gets the stop words from a given source (currently only `nltk` is supported.)

    Args:
        source:
            the source of the stop-words; currently only `nltk` is supported
        include_stop_words:
            list/set of stop-words to include
        exclude_stop_words:
            list/set of stop-words to exclude
    """
    if source == 'nltk':
        stop_words = set(nltk.corpus.stopwords.words('english'))
    else:
        raise ValueError("Only 'nlkt' source supported for stopwords.")

    if include_stop_words is not None:
        if isinstance(include_stop_words, list):
            include_stop_words = set(include_stop_words)
        stop_words |= include_stop_words

    if exclude_stop_words is not None:
        if isinstance(exclude_stop_words, list):
            exclude_stop_words = set(exclude_stop_words)
        stop_words -= exclude_stop_words

    return stop_words


def remove_stop_words(tokens: List[str],
                      stop_words: Union[List[str], Set[str]] = None) -> List[str]:
    """
    Remove stop-words from a list of tokens.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens:
            list of strings
        stop_words:
            the stop words to remove; if `None`, then remove stop-words from `nltk`.
    """
    if stop_words is None:
        stop_words = get_stop_words(source='nltk')
    elif isinstance(stop_words, list):
        stop_words = set(stop_words)

    return [t for t in tokens if t.lower() not in stop_words]


def prepare(text: str, pipeline: List[Callable] = None) -> List[str]:
    """
    Transforms `text` according to the pipeline, which is a list of functions to be called on text.

    By default (and by design), it returns a list of strings (because `tokenize()` is part of the pipeline
    if no `pipeline` is specified. However, it is possible to return something other than a list of strings
    if `tokenize()` or an equivalent is not part of the pipeline.

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


def get_n_grams(tokens: List[str],
                n: int = 2,
                separator: str = ' ',
                stop_words: Union[List[str], Set[str]] = None) -> List[str]:
    """
    Takes a list of tokens/strings and transforms that list into an n-gram list of strings.

    From:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens:
            pandas Series that is a document (text) per element
        n:
            the number of n-grams
        separator:
            string to separate the n-grams
        stop_words:
            a list or set of stop-stop
    """
    if stop_words is not None and isinstance(stop_words, list):
        stop_words = set(stop_words)

    if stop_words is None:
        stop_words = set()

    return [separator.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])
            if len([t for t in ngram if t in stop_words]) == 0]
