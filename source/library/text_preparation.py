import html
from typing import Union

import numpy as np
import pandas as pd
import regex
import textacy.preprocessing as prep
import source.library.regex_patterns as rx

import fasttext
# https://github.com/facebookresearch/fastText/issues/1067
fasttext.FastText.eprint = lambda x: None


def clean(
        text: str,
        remove_angle_bracket_content: bool = True,
        remove_bracket_content: bool = True,
        replace_urls: Union[str, None] = " _URL_ ",
        replace_hashtags: Union[str, None] = " _TAG_ ",
        replace_numbers: Union[str, None] = " _NUMBER_ ",
        replace_user_handles: Union[str, None] = " _USER_ ",
        replace_emoji: Union[str, None] = " _EMOJI_ "
        ) -> str:
    """
    Applies various functions and regex patterns to clean the string.

    This function is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

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


def predict_language(text: str,
                     probability_threshold: float = 0.6,
                     return_language_code: bool = False,
                     model_path: str = "/fasttext/lid.176.ftz",
                     model: 'fasttext' = None) -> str:
    """
    This function is a wrapper around fasttext's language prediction model. It will return the most probable
    language predicted by the model if the probabily is higher than `probability_threshold`.

    This function is modified from:
        Blueprints for Text Analytics Using Python
        by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
        (O'Reilly, 2021), 978-1-492-07408-3.
        https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb

    Args:
        text:
            the text to predict the language of
        probability_threshold:
            the minimum probability to return the language. If the probability returned by the model is lower
            than the threshold, np.nan is returned.
        return_language_code:
            if True: return the language code (e.g. en);
            If False: return the language name (e.g. English)
        model_path:
            the path of the fasttext model; if provided, the model is loaded based on the path.
        model:
            you can pass in the model directory to avoid loading each time the function is called.
    """
    # fasttext requires single line input
    text = text.replace('\n', ' ')
    if model is None:
        model = fasttext.load_model(model_path)
    labels, probabilities = model.predict(text)
    language_code = labels[0].replace("__label__", "")
    score = probabilities[0]

    if score < probability_threshold:
        return np.nan
    else:
        if return_language_code:
            return language_code
        else:
            lang_df = pd.read_csv('/code/source/resources/language_codes.csv')
            lang_df = lang_df[['name', '639-1', '639-2']].\
                melt(id_vars=['name'], var_name='iso', value_name='code')
            iso639_languages = lang_df.set_index('code')['name'].to_dict()
            return iso639_languages[language_code]
