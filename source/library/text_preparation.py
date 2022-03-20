import html
from typing import Callable, Union, List

import numpy as np
import pandas as pd
import regex
import spacy.tokens.doc
import textacy.preprocessing as prep

import source.library.regex_patterns as rx


def clean(text: str,
          remove_angle_bracket_content: bool = True,
          remove_bracket_content: bool = True,
          replace_urls: Union[str, None] = " _URL_ ",
          replace_hashtags: Union[str, None] = " _TAG_ ",
          replace_numbers: Union[str, None] = " _NUMBER_ ",
          replace_user_handles: Union[str, None] = " _USER_ ",
          replace_emoji: Union[str, None] = " _EMOJI_ ",
          ):
    """
    Applies various functions and regex patterns to clean the string.

    Args:
        text:
            text to clean
        remove_angle_bracket_content:
            If True, removes brackets and the content within the brackets (e.g. removes <this>)
            If False, removes the brackets only but leaves the content (e.g. `<this>` becomes `this`)
        remove_bracket_content:
            If True, removes brackets and the content within the brackets (e.g. removes [this])
            If False, removes the brackets only but leaves the content (e.g. `[this]` becomes `this`)
        replace_urls:
            If `replace_urls` contains a string then urls are replaced with that string value,
            (default is "_URL_"); if None, then urls are left as is.
        replace_hashtags:
            If `replace_hashtags` contains a string then hashtags are replaced with that string value,
            (default is "_TAG_"); if None, then hashtags are left as is.
        replace_numbers:
            If `replace_numbers` contains a string then numbers are replaced with that string value,
            (default is "_NUMBER_"); if None, then numbers are left as is.
        replace_user_handles:
            If `replace_user_handles` contains a string then user_handles are replaced with that string value,
            (default is "_USER_"); if None, then user_handles are left as is.
        replace_emoji:
            If `replace_emoji` contains a string then emoji are replaced with that string value,
            (default is "_EMOJI_"); if None, then emoji are left as is.

    Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
    """
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # markdown URLs like [Some text](https://....)
    text = regex.sub(rx.MARKDOWN_URLS, r' \1 ', text)

    if remove_angle_bracket_content:
        # tags like <tab>
        text = regex.sub(rx.ANGLE_BRACKETS, ' ', text)
    else:
        text = regex.sub(rx.ANGLE_BRACKETS, r' \1 ', text)
    if remove_bracket_content:
        # text or code in brackets like [0]
        text = regex.sub(rx.BRACKETS, ' ', text)
    else:
        text = regex.sub(rx.BRACKETS, r' \1 ', text)

    # standalone sequences of specials, matches &# but not #cool
    text = regex.sub(rx.STANDALONE_SPECIAL_CHARACTERS, ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = regex.sub(rx.STANDALONE_SEQUENCES, ' ', text)

    text = prep.normalize.hyphenated_words(text)
    text = prep.normalize.quotation_marks(text)
    text = prep.normalize.unicode(text)
    text = prep.remove.accents(text)

    if replace_urls is not None:
        text = prep.replace.urls(text, repl=replace_urls)

    text = prep.replace.emails(text, repl=" _EMAIL_ ")

    if replace_hashtags is not None:
        text = prep.replace.hashtags(text, repl=replace_hashtags)

    text = prep.replace.phone_numbers(text, repl=" _PHONE_ ")

    if replace_numbers is not None:
        text = prep.replace.numbers(text, repl=replace_numbers)  # needs to come after phone_numbers
    if replace_user_handles is not None:
        text = prep.replace.user_handles(text, repl=replace_user_handles)
    if replace_emoji is not None:
        text = prep.replace.emojis(text, repl=replace_emoji)

    # sequences of white spaces
    text = regex.sub(rx.WHITESPACE, ' ', text)

    return text.strip()


def create_spacy_pipeline(custom_language, custom_tokenizer: Callable):
    pass


def doc_to_dataframe(doc: spacy.tokens.doc.Doc, include_punctuation: bool = False):
    """Generate data frame for visualization of spaCy tokens."""
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


import re  ###
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, \
    compile_infix_regex, compile_suffix_regex


def custom_tokenizer(nlp):
    # use default patterns except the ones matched by re.search
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

import textacy

def extract_lemmas(doc,
                   to_lower: bool = True,
                   exclude_stopwords: bool = True,
                   exclude_punctuation: bool = True,
                   exclude_numbers: bool = False,
                   include_part_of_speech: Union[List[str], None] = None,
                   exclude_part_of_speech: Union[List[str], None] = None,
                   min_frequency: int = 1
            ):
    """

    Args:
        doc:
        to_lower:
        exclude_stopwords:
        exclude_punctuation:
        exclude_numbers:
        include_part_of_speech:  e.g. ['ADJ', 'NOUN']
        exclude_part_of_speech:
        min_frequency:
    :return:
    """
    words = textacy.extract.words(
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


def extract_noun_phrases(doc, preceding_pos=None,
                         subsequent_pos=None,
                         sep=' ', to_lower: bool = True):
    if preceding_pos is None:
        preceding_pos = ['NOUN', 'ADJ', 'VERB']
    if subsequent_pos is None:
        subsequent_pos = ['NOUN', 'ADJ', 'VERB']

    patterns = []
    for pos in preceding_pos:
        patterns.append(f"POS:{pos} POS:NOUN:+")
    for pos in subsequent_pos:
        patterns.append(f"POS:NOUN POS:{pos}:+")

    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)

    if to_lower:
        return [sep.join([t.lemma_.lower() for t in s]) for s in spans]

    return [sep.join([t.lemma_ for t in s]) for s in spans]


def extract_named_entities(doc,
                           include_types=None,
                           sep=' ',
                           to_lower=True,
                           include_label=True):
    entities = textacy.extract.entities(doc,
                                    include_types=include_types,
                                    exclude_types=None,
                                    drop_determiners=True,
                                    min_freq=1)

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

def extract_nlp(doc):
    return {
    'lemmas'          : extract_lemmas(doc,
                                     exclude_part_of_speech = ['PART', 'PUNCT',
                                        'DET', 'PRON', 'SYM', 'SPACE'],
                                     exclude_stopwords = True),
    'adjs_verbs'      : extract_lemmas(doc, include_part_of_speech = ['ADJ', 'VERB']),
    'nouns'           : extract_lemmas(doc, include_part_of_speech = ['NOUN', 'PROPN']),
    'noun_phrases'    : extract_noun_phrases(doc, ['NOUN']),
    'adj_noun_phrases': extract_noun_phrases(doc, ['ADJ']),
    'entities'        : extract_named_entities(doc, ['PERSON', 'ORG', 'GPE', 'LOC'])
    }

import fasttext

# https://github.com/facebookresearch/fastText/issues/1067
fasttext.FastText.eprint = lambda x: None


def predict_language(text, threshold=0.6, model_path = "../../lid.176.ftz", model=None):
    # fasttext requires single line input
    text = text.replace('\n', ' ')
    if model is None:
        model = fasttext.load_model(model_path)
    labels, probabilities = model.predict(text)
    language = labels[0].replace("__label__", "")
    score = probabilities[0]

    if score < threshold:
        return np.nan
    else:
        return language
