import html

import regex
import textacy.preprocessing as prep

import source.library.regex_patterns as rx


def clean(text: str):
    """
    Applies various functions and regex patterns to clean the string.

    Args:
        text:
            text to clean

    Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
    """
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = regex.sub(rx.ANGLE_BRACKETS, ' ', text)
    # markdown URLs like [Some text](https://....)
    text = regex.sub(rx.MARKDOWN_URLS, r'\1', text)
    # text or code in brackets like [0]
    text = regex.sub(rx.BRACKETS, ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = regex.sub(rx.STANDALONE_SPECIAL_CHARACTERS, ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = regex.sub(rx.STANDALONE_SEQUENCES, ' ', text)
    # sequences of white spaces
    text = regex.sub(rx.WHITESPACE, ' ', text)

    text = prep.normalize.hyphenated_words(text)
    text = prep.normalize.quotation_marks(text)
    text = prep.normalize.unicode(text)
    text = prep.remove.accents(text)

    text = prep.replace.urls(text)
    text = prep.replace.emails(text)
    text = prep.replace.hashtags(text)
    text = prep.replace.phone_numbers(text)
    text = prep.replace.numbers(text)  # needs to come after phone_numbers
    text = prep.replace.user_handles(text)
    text = prep.replace.emojis(text)

    return text.strip()
