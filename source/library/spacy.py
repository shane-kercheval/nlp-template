from typing import Callable, Collection, Optional, Union, List

import pandas as pd
import spacy.tokens.doc
from spacy.language import Language

import re
import textacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


class SpacyWrapper:
    def __init__(
            self,
            stopwords_to_add: Union[set[str], None] = None,
            stopwords_to_remove: Union[set[str], None] = None,
            tokenizer: Callable = None):
        self._nlp = spacy.load('en_core_web_sm')

        # https://machinelearningknowledge.ai/tutorial-for-stopwords-in-spacy/#i_Stopwords_List_in_Spacy
        if stopwords_to_add is not None:
            if isinstance(stopwords_to_add, list):
                stopwords_to_add = set(stopwords_to_add)
            self._nlp.Defaults.stop_words |= stopwords_to_add

        if stopwords_to_remove is not None:
            if isinstance(stopwords_to_remove, list):
                stopwords_to_remove = set(stopwords_to_remove)
            self._nlp.Defaults.stop_words -= stopwords_to_remove

        if tokenizer is None:
            self._nlp.tokenizer = custom_tokenizer(self._nlp)
        else:
            self._nlp.tokenizer = tokenizer(self._nlp)

    @property
    def stop_words(self) -> set[str]:
        return self._nlp.Defaults.stop_words

    def extract(
            self,
            documents: Collection[str],
            all_lemmas: bool = True,
            partial_lemmas: bool = True,
            bi_grams: bool = True,
            adjectives_verbs: bool = True,
            nouns: bool = True,
            noun_phrases: bool = True,
            named_entities: bool = True) -> dict[list[list[str]]]:
        docs = self._nlp.pipe(documents)

        extracted_values = [None] * len(documents)
        for j, doc in enumerate(docs):
            extracted_values[j] = extract_from_doc(
                doc=doc,
                stop_words=self.stop_words,
                all_lemmas=all_lemmas,
                partial_lemmas=partial_lemmas,
                bi_grams=bi_grams,
                adjectives_verbs=adjectives_verbs,
                nouns=nouns,
                noun_phrases=noun_phrases,
                named_entities=named_entities,
            )

        return _list_dicts_to_dict_lists(list_of_dicts=extracted_values)


def _list_dicts_to_dict_lists(list_of_dicts: list[dict]):
    """
    Args:
        list_of_dicts:
            e.g. `[{'x': 1, 'y': 10}, {'x': 2, 'y': 11}, {'x': 3, 'y': 12}]`
    Returns:
        e.g. `{'x': [1, 2, 3], 'y': [10, 11, 12]}`
    """
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            dict_of_lists.setdefault(key, []).append(value)

    return dict_of_lists


list_of_dicts = [{'x': 1, 'y': 10}, {'x': 2, 'y': 11}, {'x': 3, 'y': 12}]
expected = {'x': [1, 2, 3], 'y': [10, 11, 12]}
assert _list_dicts_to_dict_lists(list_of_dicts) == expected

list_of_dicts = [{'y': 10}, {'x': 2, 'y': 11}, {'x': 3}, {'x': 4}]
expected = {'x': [2, 3, 4], 'y': [10, 11]}
assert _list_dicts_to_dict_lists(list_of_dicts) == expected


def doc_to_dataframe(doc: spacy.tokens.doc.Doc, include_punctuation: bool = False) -> pd.DataFrame:
    """
    This code takes a spaCy Doc and converts the doc into a pd.DataFrame

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to convert
        include_punctuation: whether or not to return the punctuation in the DataFrame.
    """
    rows = []
    for i, t in enumerate(doc):
        if (not t.is_punct and t.pos_ != 'PUNCT') or include_punctuation:
            row = {
                'token': i, 'text': t.text, 'lemma_': t.lemma_,
                'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
                'pos_': t.pos_, 'dep_': t.dep_,
                'ent_type_': t.ent_type_, 'ent_iob_': t.ent_iob_
            }
            rows.append(row)

    df = pd.DataFrame(rows).set_index('token')
    df.index.name = None
    return df


def custom_tokenizer(nlp: Language):
    """
    This code creates a custom tokenizer, as described on pg. 108 of

        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        nlp: the Language
    """
    prefixes = [
        pattern for pattern in nlp.Defaults.prefixes if pattern not in ['-', '_', '#']
    ]
    suffixes = [
        pattern for pattern in nlp.Defaults.suffixes if pattern not in ['_']
    ]
    infixes = [
        pattern for pattern in nlp.Defaults.infixes if not re.search(pattern, 'xx-xx')
    ]

    return Tokenizer(
        vocab=nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(prefixes).search,
        suffix_search=compile_suffix_regex(suffixes).search,
        infix_finditer=compile_infix_regex(infixes).finditer,
        token_match=nlp.Defaults.token_match
    )


def extract_lemmas(doc: spacy.tokens.doc.Doc,
                   exclude_stopwords: bool = True,
                   stop_words: Optional[set] = None,
                   exclude_punctuation: bool = True,
                   exclude_numbers: bool = False,
                   include_part_of_speech: Union[List[str], None] = None,
                   exclude_part_of_speech: Union[List[str], None] = None,
                   min_frequency: int = 1) -> list[str]:
    """
    This function extracts lemmas from the `doc`.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        exclude_stopwords: if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        exclude_punctuation: if True, exclude punctuation
        exclude_numbers: if True, exclude numbers
        include_part_of_speech:  e.g. ['ADJ', 'NOUN']
        exclude_part_of_speech: e.g. ['ADJ', 'NOUN']
        min_frequency: the minimum frequency required for the lemma
    """
    words = textacy.extract.words(  # noqa
        doc,
        filter_stops=exclude_stopwords,
        filter_punct=exclude_punctuation,
        filter_nums=exclude_numbers,
        include_pos=include_part_of_speech,
        exclude_pos=exclude_part_of_speech,
        min_freq=min_frequency,
    )
    tokens = [token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in words]
    if exclude_stopwords and stop_words:
        # exclude_stopwords appears to remove pre-lemmatized stop-words
        return [t for t in tokens if t not in stop_words]

    return tokens


def extract_n_grams(doc: spacy.tokens.doc.Doc,
                    n=2,
                    sep: str = ' ',
                    exclude_stopwords: bool = True,
                    stop_words: Optional[set] = None,
                    exclude_punctuation: bool = True,
                    exclude_numbers: bool = False,
                    include_part_of_speech: Union[List[str], None] = None,
                    exclude_part_of_speech: Union[List[str], None] = None,
                    ):
    """
    This function extracts n_grams from the `doc`.

    Note that exclude_punctuation and exclude_numbers does not seem to work; punctuation and
    numbers are being returned.


    Args:
        doc: the doc to extract from
        n: the number of grams to return
        sep: the string that will separate the n grams.
        exclude_stopwords: if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        exclude_punctuation: if True, exclude punctuation
        exclude_numbers: if True, exclude numbers
        include_part_of_speech:  e.g. ['ADJ', 'NOUN']
        exclude_part_of_speech: e.g. ['ADJ', 'NOUN']
    """
    spans = textacy.extract.basics.ngrams(  # noqa
        doc,
        n=n,
        filter_stops=exclude_stopwords,
        filter_punct=exclude_punctuation,
        filter_nums=exclude_numbers,
        include_pos=include_part_of_speech,
        exclude_pos=exclude_part_of_speech,
    )
    spans = list(spans)
    n_grams = [
        sep.join([token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in s])
        for s in spans
    ]
    if exclude_stopwords and stop_words:
        has_stopwords = [any([t.lemma_.lower() in stop_words for t in s]) for s in spans]
        return [gram for stop, gram in zip(has_stopwords, n_grams) if not stop]

    return n_grams


def extract_noun_phrases(doc: spacy.tokens.doc.Doc,
                         exclude_stopwords: bool = True,
                         stop_words: Optional[set] = None,
                         preceding_part_of_speech: Union[List[str], None] = None,
                         subsequent_part_of_speech: Union[List[str], None] = None,
                         sep: str = ' '):
    """
    This function extracts the "noun phrases" from `doc` and returns the lemmas.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc:
            the doc to extract from
        exclude_stopwords:
            if True, exclude stopwords
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        preceding_part_of_speech:
            Part of Speech to filter for in the preceding word. If None, default is ['NOUN', 'ADJ',
            'VERB']
        subsequent_part_of_speech:
            Part of Speech to filter for in the subsequent word. If None, default is ['NOUN',
            'ADJ', 'VERB']
        sep:
            the separator to join the lemmas on.
    """
    if preceding_part_of_speech is None:
        preceding_part_of_speech = ['NOUN', 'ADJ', 'VERB']
    if subsequent_part_of_speech is None:
        subsequent_part_of_speech = ['NOUN', 'ADJ', 'VERB']

    patterns = []
    for pos in preceding_part_of_speech:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    for pos in subsequent_part_of_speech:
        patterns.append(f"POS:NOUN POS:{pos}:+")

    spans = textacy.extract.matches.token_matches(
        doc,
        patterns=patterns,
    )
    spans = list(spans)
    phrases = [
        sep.join([token if (token := t.lemma_.lower()) != 'datum' else 'data' for t in s])
        for s in spans
    ]
    if exclude_stopwords and stop_words:
        # Spacy doesn't remove stopwords when the lemma itself is a stopword so we have to manually
        # remove
        has_stopwords = [any([t.lemma_.lower() in stop_words for t in s]) for s in spans]
        return [phrase for stop, phrase in zip(has_stopwords, phrases) if not stop]

    return phrases


def extract_named_entities(doc: spacy.tokens.doc.Doc,
                           include_types: Union[List[str], None] = None,
                           sep: str = ' ',
                           include_label: bool = True):
    """
    This function extracts the named entities from the doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        include_types:  Part of Speech to filter to include.
        sep: the separator to join the lemmas on.
        include_label: if True, include the type (i.e. label) of the named entity.
    """
    entities = textacy.extract.entities(  # noqa
        doc,
        include_types=include_types,
        exclude_types=None,
        drop_determiners=True,
        min_freq=1
    )

    def format_named_entity(entity):
        lemmas = sep.join([t.lemma_ for t in entity])

        value = lemmas
        if include_label:
            value = f'{value} ({entity.label_})'
        return value

    return [format_named_entity(e) for e in entities]


def extract_from_doc(doc: spacy.tokens.doc.Doc,
                     stop_words: Optional[set] = None,
                     all_lemmas: bool = True,
                     partial_lemmas: bool = True,
                     bi_grams: bool = True,
                     adjectives_verbs: bool = True,
                     nouns: bool = True,
                     noun_phrases: bool = True,
                     named_entities: bool = True) -> dict:
    """
    This function extracts common types of tokens from a spaCy doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc: the doc to extract from
        stop_words:
            list of stop-words; needed to remove lemmas that are stop words, spacy does not seem
            to do this.
        all_lemmas: if True, return all lemmas (lower-case) of every word, also includes
        punctuation;
            This is useful if you need to use e.g. scikit-learn TfidfVectorizer and need the raw
            document. i.e. you can do a `' '.join()` with this data.
        partial_lemmas: if True, return lemmas (lower-case), excluding stopwords and punctuation
        bi_grams: if True, return bi_grams
        adjectives_verbs: if True, return adjectives_verbs
        nouns: if True, return nouns
        noun_phrases: if True, return noun_phrases
        named_entities: if True, return named_entities
    """
    results = dict()
    if all_lemmas:
        results['all_lemmas'] = extract_lemmas(
            doc,
            stop_words=None,
            exclude_stopwords=False,
            exclude_punctuation=True,
            exclude_numbers=False,
            include_part_of_speech=None,
            exclude_part_of_speech=None,
            min_frequency=1
        )
    if partial_lemmas:
        results['partial_lemmas'] = extract_lemmas(
            doc,
            exclude_part_of_speech=['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if bi_grams:
        results['bi_grams'] = extract_n_grams(
            doc,
            n=2,
            sep='-',
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if adjectives_verbs:
        results['adjs_verbs'] = extract_lemmas(
            doc,
            include_part_of_speech=['ADJ', 'VERB'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if nouns:
        results['nouns'] = extract_lemmas(
            doc,
            include_part_of_speech=['NOUN', 'PROPN'],
            exclude_stopwords=True,
            stop_words=stop_words,
        )
    if noun_phrases:
        # Setting exclude_stopwords to true is extemely slow
        # need to figure this out if we decide to use noun-phrases
        results['noun_phrases'] = extract_noun_phrases(
            doc,
            exclude_stopwords=True,
            stop_words=stop_words,
            sep='-'
        )
    if named_entities:
        results['entities'] = extract_named_entities(doc)

    return results
