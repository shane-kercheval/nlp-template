from collections import Counter
from typing import List, Union, Set

import numpy as np
import pandas as pd
import regex
import regex as re
from textacy.extract.kwic import keyword_in_context
import source.library.regex_patterns as rx


def count_tokens(tokens: Union[pd.Series, List[list], List[str]],
                 min_frequency: int = 2,  # noqa
                 count_once_per_doc: bool = False,
                 remove_tokens: Union[Set[str], None] = None) -> pd.DataFrame:
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
        remove_tokens:
            remove the set of tokens from the results.

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

    if remove_tokens is not None:
        if isinstance(remove_tokens, str):
            remove_tokens = {remove_tokens}
        freq_df = freq_df[~freq_df.index.isin(remove_tokens)]

    return freq_df.sort_values('frequency', ascending=False)


def count_text_patterns(documents: Union[pd.Series, List[str], str],
                        pattern: str,
                        min_frequency: int = 1  # noqa
                        ) -> pd.DataFrame:
    # counts all matches of `regex` across all `documents`
    # documents is either Series of strings (documents)
    # or list of strings, a single string
    """
    Counts all matches of the regex `pattern` across all `documents`.

    Modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        documents:
            either
                - a pandas Series (with strings); each item in the Series is a 'document'
                - a list of strings; each string item in list is a 'document'
                - a single string; which represents a single 'document'
        pattern:
            The regex containing the pattern to search for.
        min_frequency:
            The minimum times the pattern has to appear in order to be returned
    """
    if isinstance(documents, str):
        documents = pd.Series([documents])
    if isinstance(documents, list):
        documents = pd.Series(documents)

    # create counter and run through all data
    counter = Counter()

    def count(text: str):
        counter.update(regex.findall(pattern, text))

    _ = documents.map(count)

    # transform counter into data frame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['frequency'])
    freq_df = freq_df.query('frequency >= @min_frequency')
    freq_df.index.name = 'match'

    return freq_df.sort_values('frequency', ascending=False)


def term_frequency(df: pd.DataFrame,
                   tokens_column: str = None,
                   segment_columns: Union[str, List[str]] = None,
                   min_frequency: int = 2,
                   remove_tokens: Union[Set[str], None] = None) -> pd.DataFrame:
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
        remove_tokens:
            remove the set of tokens from the results.
    """
    if segment_columns is None:
        term_freq = count_tokens(df[tokens_column], min_frequency=min_frequency, remove_tokens=remove_tokens)
    else:
        term_freq = df.groupby(segment_columns)[tokens_column].\
            apply(count_tokens, min_frequency=min_frequency, remove_tokens=remove_tokens)

    return term_freq


def inverse_document_frequency(documents: Union[pd.Series, List[list]],
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
           segment_columns: Union[str, List[str]] = None,
           min_frequency_document: int = 1,
           min_frequency_corpus: int = 2,
           remove_tokens: Union[Set[str], None] = None,
           constant: float = 0.1) -> pd.DataFrame:
    """
    This function returns the term-frequency inverse-document-frequency values for a given set of documents,
    or by segment/slice.

    Unlike sci-kit learn TfidfVectorizer, you can calculate tf-idf for different slices/segments.

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
        remove_tokens:
            remove the set of tokens from the results.
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
        segment_columns=segment_columns,
        remove_tokens=remove_tokens
    )
    inverse_doc_freq = inverse_document_frequency(
        documents=df[tokens_column],
        min_frequency=min_frequency_corpus,
        constant=constant
    )
    tf_idf_df = term_freq['frequency'] * inverse_doc_freq['inverse document frequency']
    tf_idf_df.dropna(inplace=True)
    tf_idf_df.name = 'tf-idf'

    result = term_freq.join(pd.DataFrame(tf_idf_df.sort_values(ascending=False)), how='inner')
    assert result.shape[0] == len(tf_idf_df)
    sort_by = []
    ascending = []
    if segment_columns is not None:
        if isinstance(segment_columns, str):
            segment_columns = [segment_columns]
        sort_by = segment_columns
        ascending = [True] * len(sort_by)
    sort_by += ['tf-idf', 'token']
    ascending += [False, True]
    result.sort_values(ascending=ascending, by=sort_by, inplace=True)

    return result


def get_context_from_keyword(documents: pd.Series,
                             keyword: str,
                             num_samples: int = 10,
                             window_width: int = 35,
                             keyword_wrap: str = '|',
                             pad_context: bool = False,
                             ignore_case: bool = True,
                             shuffle_data: bool = True,
                             random_seed: int = None) -> List[str]:
    """
    Search documents for specific keyword and return matches/context in list.

    Wrapper around https://textacy.readthedocs.io/en/0.11.0/_modules/textacy/extract/kwic.html
    Args:
        documents:
            pandas Series that is a document (text) per element
        keyword:
            keyword to search for
        num_samples:
            the number of samples to return
        window_width:
            Number of characters on either side of ``keyword`` to include as "context".
        keyword_wrap:
            Character/text to prepend/append (i.e. wrap around) the keyword
        pad_context:
            pad_context: If True, pad pre- and post-context strings to ``window_width``
            chars in length; otherwise, us as many chars as are found in the text,
            up to the specified width.
        ignore_case: If True, ignore letter case in ``keyword`` matching; otherwise,
            use case-sensitive matching. Note that this argument is only used if
            ``keyword`` is a string; for pre-compiled regular expressions,
            the ``re.IGNORECASE`` flag is left as-is.
        shuffle_data:
            If True, shuffle the data before searching for context, in order to generate a random sample.
        random_seed:
            Random state if shuffle_data is True.
    """
    num_contexts_found = 0
    contexts_found = []

    if shuffle_data:
        documents = documents.sample(len(documents), random_state=random_seed)
    # iterate through documents and generator until the number of samples is found
    # the list could get quite large and we could run out of memory if we converted the generators
    # to lists and built one large list.
    for doc in documents:
        context_gen = keyword_in_context(
            doc=doc,
            keyword=keyword,
            ignore_case=ignore_case,
            window_width=window_width,
            pad_context=pad_context)
        for context in context_gen:
            before = re.sub(r'[\n\t]', ' ', context[0])
            after = re.sub(r'[\n\t]', ' ', context[2])
            contexts_found += [f"{before} {keyword_wrap}{context[1]}{keyword_wrap} {after}"]
            num_contexts_found += 1
            if num_contexts_found >= num_samples:
                return contexts_found

    return contexts_found


def count_keywords(tokens: Union[List[str], Set[str]],
                   keywords: Union[List[str], Set[str]]) -> List[int]:
    """
    This function counts the number of times each keyword appears in the list of tokens.
    It returns a list the same length as keywords and the number of occurrences, for each respective
    keyword (in order of the keywords passed in).

    Copied from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        tokens:
            tokens from a single document; i.e. a list of strings/tokens
        keywords:
            a list of keywords to get counts for
    """
    tokens = [t for t in tokens if t in keywords]
    counter = Counter(tokens)
    return [counter.get(k, 0) for k in keywords]


def count_keywords_by(df: pd.DataFrame,
                      by: Union[str, List[str]],
                      tokens: str,
                      keywords: List[str],
                      count_once_per_doc: bool = False) -> pd.DataFrame:
    """
    This function is used to count the number of times keywords are used across one or more groups (e.g.
    across `year`, or across `year and country`, etc.).

    This function takes a dataframe that has a column containing tokens (the column name is passed to
    `tokens`) and additional columns to group by (i.e. `by`).

    Copied from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch01/First_Insights.ipynb

    Args:
        df:
            the data.frame
        by:
            the column(s) to group by
        tokens:
            the column name in `df` containing the tokens
        keywords:
            a list of keywords to get counts for
        count_once_per_doc:
            If False, the counts contain the total number of occurrences summed across all documents for the
                given group (for each keyword).
            If True, each keyword count is only counted once per once per document. In other words, it tells
                you the amount of documents the keyword appears in for the given groups.
    """
    def count_keywords_adjusted(x, keywords):  # noqa
        if count_once_per_doc:
            return count_keywords(tokens=set(x), keywords=keywords)
        else:
            return count_keywords(tokens=x, keywords=keywords)

    df = df.reset_index(drop=True)  # if the supplied dataframe has gaps in the index
    freq_matrix = df[tokens].apply(count_keywords_adjusted, keywords=keywords)
    freq_df = pd.DataFrame.from_records(freq_matrix, columns=keywords)
    freq_df[by] = df[by]  # copy the grouping column(s)

    return freq_df.groupby(by=by).sum().sort_values(by)


def impurity(text: str, pattern: str = rx.SUSPICIOUS, min_length: int = 10):
    """
    Returns the percent of characters matching regex_patterns.SUSPICIOUS (or other pattern passed in).

    Args:
        text:
            text to search
        pattern:
            regex pattern to use to search for suspicious characters; default is regex_patterns.SUSPICIOUS
        min_length:
            the minimum length of `text` required to return a score; if `text` is less than the minimum
            length, then a value of np.nan is returned.
    Copied from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
         https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
    """
    if text is None or len(text) < min_length:
        return np.nan
    else:
        return len(regex.findall(pattern, text))/len(text)
