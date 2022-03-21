from typing import Callable, Union, List

import pandas as pd
import spacy.tokens.doc
from spacy.language import Language

import re
import textacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


def create_spacy_pipeline(language=None, tokenizer: Callable = None) -> Language:
    """
    This code creates a spacy pipeline.
    
    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        language: a spaCy language (e.g. see spacy_language.py)
        tokenizer: a custom tokenizer function
    :return:
    """
    if language is None:
        nlp = spacy.load('en_core_web_sm', disable=[])
    else:
        nlp = language()

    if tokenizer is not None:
        nlp.tokenizer = tokenizer(nlp)

    return nlp


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
        if not t.is_punct or include_punctuation:
            row = {'token': i, 'text': t.text, 'lemma_': t.lemma_,
                   'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
                   'pos_': t.pos_, 'dep_': t.dep_,
                   'ent_type_': t.ent_type_, 'ent_iob_': t.ent_iob_}
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
    prefixes = [pattern for pattern in nlp.Defaults.prefixes
                if pattern not in ['-', '_', '#']]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes
                if pattern not in ['_']]
    infixes = [pattern for pattern in nlp.Defaults.infixes
               if not re.search(pattern, 'xx-xx')]

    return Tokenizer(vocab=nlp.vocab,  # noqa
                     rules=nlp.Defaults.tokenizer_exceptions,  # noqa
                     prefix_search=compile_prefix_regex(prefixes).search,  # noqa
                     suffix_search=compile_suffix_regex(suffixes).search,  # noqa
                     infix_finditer=compile_infix_regex(infixes).finditer,  # noqa
                     token_match=nlp.Defaults.token_match)  # noqa


def extract_lemmas(doc: spacy.tokens.doc.Doc,
                   to_lower: bool = True,
                   exclude_stopwords: bool = True,
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
        to_lower: if True, call `lower()` on the lemma
        exclude_stopwords: if True, exclude stopwords
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
    if to_lower:
        return [t.lemma_.lower() for t in words]

    return [t.lemma_ for t in words]


def extract_noun_phrases(doc: spacy.tokens.doc.Doc,
                         preceding_part_of_speech: Union[List[str], None] = None,
                         subsequent_part_of_speech: Union[List[str], None] = None,
                         sep: str = ' ',
                         to_lower: bool = True):
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
        preceding_part_of_speech: 
            Part of Speech to filter for in the preceding word. If None, default is ['NOUN', 'ADJ', 'VERB']
        subsequent_part_of_speech:
            Part of Speech to filter for in the subsequent word. If None, default is ['NOUN', 'ADJ', 'VERB']
        sep:
            the separator to join the lemmas on.
        to_lower:
            if True, call `lower()` on the lemma
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

    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)  # noqa

    if to_lower:
        return [sep.join([t.lemma_.lower() for t in s]) for s in spans]

    return [sep.join([t.lemma_ for t in s]) for s in spans]


def extract_named_entities(doc: spacy.tokens.doc.Doc,
                           include_types: Union[List[str], None] = None,
                           sep: str = ' ',
                           include_label: bool = True,
                           to_lower: bool = True):
    """
    This function extracts the named entities from the doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        doc:
            the doc to extract from
        include_types: 
            Part of Speech to filter to include.
        sep:
            the separator to join the lemmas on.
        include_label:
            if True, include the type (i.e. label) of the named entity.
        to_lower:
            if True, call `lower()` on the lemma
    """
    entities = textacy.extract.entities(  # noqa
        doc,
        include_types=include_types,
        exclude_types=None,
        drop_determiners=True,
        min_freq=1
    )

    def format_named_entity(entity):
        if to_lower:
            lemmas = sep.join([t.lemma_.lower() for t in entity])
        else:
            lemmas = sep.join([t.lemma_ for t in entity])

        value = lemmas
        if include_label:
            value = f'{value} ({entity.label_})'
        return value

    return [format_named_entity(e) for e in entities]


def extract_from_doc(doc: spacy.tokens.doc.Doc,
                     include_lemmas: bool = True,
                     include_adjectives_verbs: bool = True,
                     include_nouns: bool = True,
                     include_noun_phrases: bool = True,
                     include_adject_none_phrases: bool = True,
                     include_entities: bool = True) -> dict:
    """
    This function extracts common types of tokens from a spaCy doc.

    This code is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    """
    results = dict()
    if include_lemmas:
        results['lemmas'] = extract_lemmas(
            doc,
            exclude_part_of_speech=['PART', 'PUNCT', 'DET', 'PRON', 'SYM', 'SPACE'],
            exclude_stopwords=True
        )
    if include_adjectives_verbs:
        results['adjs_verbs'] = extract_lemmas(doc, include_part_of_speech=['ADJ', 'VERB'])
    if include_nouns:
        results['nouns'] = extract_lemmas(doc, include_part_of_speech=['NOUN', 'PROPN'])
    if include_noun_phrases:
        results['noun_phrases'] = extract_noun_phrases(doc)
    if include_adject_none_phrases:
        results['adj_noun_phrases'] = extract_noun_phrases(doc, ['ADJ'])
    if include_entities:
        results['entities'] = extract_named_entities(doc)

    return results
