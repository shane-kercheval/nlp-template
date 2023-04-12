import fasttext
import pandas as pd
from source.library.text_preparation import clean, predict_language
from tests.helpers import dataframe_to_text_file, get_test_file_path


def test__clean(reddit):
    text = "<This> 😩😬 [sentence] [sentence]& a & [link to something](www.hotmail.com) " \
        "stuff && abc --- -   https://www.google.com/search?q=asdfa; john.doe@gmail.com " \
        "#remove 445 583.345.7833 @shane"
    clean_text = clean(text)
    expected_text = '_EMOJI_ _EMOJI_ a link to something stuff abc - _URL_ _EMAIL_ _TAG_ remove ' \
        '_NUMBER_ _PHONE_ _USER_'
    assert expected_text == clean_text

    clean_text = clean(
        text,
        remove_angle_bracket_content=False,
        remove_bracket_content=False,
        replace_urls=None,
        replace_hashtags=None,
        replace_numbers=None,
        replace_user_handles=None,
        replace_emoji=None,
    )
    expected_text = 'This 😩😬 sentence sentence a link to something stuff abc - ' \
        'https://www.google.com/search?q=asdfa; _EMAIL_ #remove 445 _PHONE_ @shane'
    assert expected_text == clean_text

    with open(get_test_file_path('text_preparation/clean__reddit.txt'), 'w') as handle:
        handle.writelines([x + "\n" for x in reddit['post'].apply(clean)])

    with open(get_test_file_path('text_preparation/example_unclean.txt'), 'r') as handle:
        text_lines = handle.readlines()

    with open(get_test_file_path('text_preparation/example_clean.txt'), 'w') as handle:
        handle.writelines([clean(x) + "\n" for x in text_lines])


def test__predict_language():
    model = fasttext.load_model("/fasttext/lid.176.ftz")
    assert predict_language('This is english.', model=model, return_language_code=False) == "English"  # noqa
    assert predict_language('This is english.', model=model, return_language_code=True) == "en"

    language_text = [
        "I don't like version 2.0 of Chat4you 😡👎",  # English
        "Ich mag Version 2.0 von Chat4you nicht 😡👎",  # German
        "Мне не нравится версия 2.0 Chat4you 😡👎",  # Russian
        "Não gosto da versão 2.0 do Chat4you 😡👎",  # Portuguese
        "मुझे Chat4you का संस्करण 2.0 पसंद नहीं है 😡👎"]  # Hindi
    language_text_df = pd.Series(language_text, name='text').to_frame()

    # create new column
    language_text_df['language'] = language_text_df['text']\
        .apply(predict_language, model=model)
    language_text_df['language_code'] = language_text_df['text'].apply(
        predict_language,
        model=model,
        return_language_code=True
    )
    dataframe_to_text_file(
        language_text_df,
        get_test_file_path('text_preparation/predict_language.txt')
    )
