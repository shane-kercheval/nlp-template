from typing import List, Callable

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