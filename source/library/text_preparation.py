import html
from typing import Callable, Union

import pandas as pd
import regex
import spacy.tokens.doc
import textacy.preprocessing as prep
import textacy.preprocessing.resources as prep_resources

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
        replace_emails:
            If `replace_emails` contains a string then emails are replaced with that string value,
            (default is "_EMAIL_"); if None, then emails are left as is.
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


def doc_to_dataframe(spacy_doc: spacy.tokens.doc.Doc, include_punctuation: bool = False):
    """Generate data frame for visualization of spaCy tokens."""
    rows = []
    for i, t in enumerate(spacy_doc):
        if not t.is_punct or include_punctuation:
            row = {'token': i, 'text': t.text, 'lemma_': t.lemma_,
                   'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
                   'pos_': t.pos_, 'dep_': t.dep_,
                   'ent_type_': t.ent_type_, 'ent_iob_': t.ent_iob_}
            rows.append(row)

    df = pd.DataFrame(rows).set_index('token')
    df.index.name = None
    return df

