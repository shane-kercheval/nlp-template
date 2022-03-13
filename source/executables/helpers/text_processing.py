from collections import Counter
from typing import List, Callable, Union
import nltk
import numpy as np
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


def count_tokens(tokens: Union[pd.Series, list], min_frequency: int = 2, count_once_per_doc=False) -> pd.DataFrame:  # noqa
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
                - a pandas Series (with strings or a list of strings); each item in the Series is a 'document'
                - a list of strings; a single 'document'
                - a list of lists of strings; each sublist is a 'document'
        min_frequency:
            The minimum times the token has to appear in order to be returned
        count_once_per_doc:
            If True, counts each token once per document. This is synonymous with 'Document Frequency'.

            e.g. True:
                [['a', 'b', 'b'], ['b', 'c', 'c']]

                'b' | 2
                'a' | 1
                'c' | 1

            e.g. False:
                 [['a', 'b', 'b'], ['b', 'c', 'c']]

                'b' | 3
                'c' | 2
                'a' | 1
    """
    if isinstance(tokens, list):
        if isinstance(tokens[0], list):
            tokens = pd.Series(tokens)
        else:
            tokens = pd.Series([tokens])

    # create counter and run through all data
    counter = Counter()

    def count(x):
        # if we pass a string it will count characters, which is not the intent
        if isinstance(x, str):
            x = [x]
        if count_once_per_doc:
            x = set(x)
        counter.update(x)

    _ = tokens.map(count)

    # transform counter into data frame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['frequency'])
    freq_df = freq_df.query('frequency >= @min_frequency')
    freq_df.index.name = 'token'
    
    return freq_df.sort_values('frequency', ascending=False)


def term_frequency(df: pd.DataFrame,
                   tokens_column: str = None,
                   segment_columns: Union[str, list] = None,
                   min_frequency: int = 2) -> pd.DataFrame:
    """
    This function is similar to `count_tokens()`, but calculates term-frequency across different
    segments/slices.

    Args:
        df:
            dataframe with columns of tokens, and optionally a column to slice/segment by.
        tokens_column:
            the name of the column containing the tokens.
        segment_columns:
            the name(s) of the column(s) to segment term-frequencies by. Either a string or a list of strings.
        min_frequency:
            The minimum times the token has to appear in order to be returned
    """
    if segment_columns is None:
        term_freq = count_tokens(df[tokens_column], min_frequency=min_frequency)
    else:
        term_freq = df.groupby(segment_columns)[tokens_column].apply(count_tokens, min_frequency=min_frequency)

    return term_freq


def inverse_document_frequency(documents: Union[pd.Series, list],
                               min_frequency: int = 2,
                               constant: float = 0.1) -> pd.DataFrame:
    """
    This function calculates the inverse document frequency (IDF)

    Modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        documents:
            either a Pandas series, each element containing a list of tokens/words;
            or a list of lists (of tokens/words)
        min_frequency:
            The minimum times the token has to appear in order to be returned
        constant:
            "Note that idf(t)=0 for terms that appear in all documents. To not completely ignore those term,
            some libraries add a constant to the whole term." - Blueprints for Text Analytics Using Python,
            pg. 21
    """
    document_frequency = count_tokens(tokens=documents, min_frequency=min_frequency, count_once_per_doc=True)
    num_documents = len(documents)
    # if any frequencies are greater than the number of documents, something is wrong, because we are only
    # counting once per document
    assert (document_frequency['frequency'] <= num_documents).all()
    idf = np.log(num_documents / document_frequency) + constant
    idf.rename(columns={'frequency': 'inverse document frequency'}, inplace=True)
    return idf


def tf_idf(df: pd.DataFrame,
           tokens_column: str = None,
           segment_columns: Union[str, list] = None,
           min_frequency_document: int = 1,
           min_frequency_corpus: int = 2,
           constant: float = 0.1) -> pd.DataFrame:
    """
    This function returns the term-frequency inverse-document-frequency values for a given set of documents,
    or by segment/slice.

    Args:
        df:
            dataframe with columns of tokens, and optionally a column to slice/segment by.
        tokens_column:
            the name of the column containing the tokens.
        segment_columns:
            the name(s) of the column(s) to segment term-frequencies by. Either a string or a list of strings.
        min_frequency_document:
            The minimum times the token has to appear in the document in order to be returned
        min_frequency_corpus:
            The minimum times the token has to appear in the corpus (across all docs) in order to be returned
        constant:
            "Note that idf(t)=0 for terms that appear in all documents. To not completely ignore those term,
            some libraries add a constant to the whole term." - Blueprints for Text Analytics Using Python,
            pg. 21
    """
    # calculate the term-frequencies by segment (if applicable) but inverse-document-frequencies across
    # entire dataset

    term_freq = term_frequency(
        df=df,
        tokens_column=tokens_column,
        min_frequency=min_frequency_document,
        segment_columns=segment_columns
    )
    inverse_doc_freq = inverse_document_frequency(
        documents=df[tokens_column],
        min_frequency=min_frequency_corpus,
        constant=constant
    )
    tf_idf_df = term_freq['frequency'] * inverse_doc_freq['inverse document frequency']
    tf_idf_df.dropna(inplace=True)
    tf_idf_df.name='tf-idf'

    result = term_freq.join(pd.DataFrame(tf_idf_df.sort_values(ascending=False)), how='inner')
    assert result.shape[0] == len(tf_idf_df)
    sort_by = []
    ascending = []
    if segment_columns is not None:
        if isinstance(segment_columns, str):
            segment_columns = [segment_columns]
        sort_by = segment_columns
        ascending = [True] * len(sort_by)
    sort_by += ['tf-idf']
    ascending += [False]
    result.sort_values(ascending=ascending, by=sort_by, inplace=True)
    return result





