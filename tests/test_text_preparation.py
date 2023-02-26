import fasttext
import numpy as np
import pandas as pd
import spacy
# from spacy.lang.en import English

from source.library.spacy import SpacyWrapper, doc_to_dataframe, custom_tokenizer, extract_lemmas, \
    extract_noun_phrases, extract_named_entities, extract_from_doc, extract_n_grams
from source.library.text_preparation import clean, predict_language
from tests.helpers import get_test_file_path, dataframe_to_text_file


def test__clean(reddit):
    text = "<This> 😩😬 [sentence] [sentence]& a & [link to something](www.hotmail.com) " \
        "stuff && abc --- -   https://www.google.com/search?q=asdfa; john.doe@gmail.com " \
        "#remove 445 583.345.7833 @shane"
    clean_text = clean(text)
    expected_text = '_EMOJI_ _EMOJI_ a link to something stuff abc - _URL_ _EMAIL_ _TAG_ ' \
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


def test__get_stopwords():
    stop_words = SpacyWrapper().stop_words
    assert len(stop_words) > 300
    assert 'is' in stop_words


def test__doc_to_dataframe():
    text = "This: _is_ _some_ #text down dear regards."
    nlp = SpacyWrapper()._nlp
    assert isinstance(nlp, spacy.lang.en.English)
    # if we do nlp() rather than make_doc, we should still get lemmas/etc.
    doc = nlp(text)
    df = doc_to_dataframe(doc)
    assert not df.drop(columns=['ent_type_', 'ent_iob_']).replace('', np.nan).isna().any().any()
    default_tokens = [str(x) for x in doc]
    assert default_tokens == [
        'This', ':', '_is_', '_some_', '#text', 'down', 'dear', 'regards', '.'
    ]
    default_df = doc_to_dataframe(doc, include_punctuation=True)
    assert default_df.query("text == 'down'").is_stop.iloc[0]
    assert not default_df.query("text == 'dear'").is_stop.iloc[0]
    assert not default_df.query("text == 'regards'").is_stop.iloc[0]
    assert (default_df['pos_'] == 'PUNCT').any()
    assert not (doc_to_dataframe(doc, include_punctuation=False)['pos_'] == 'PUNCT').any()

    nlp = SpacyWrapper(
        stopwords_to_add={'dear', 'regards'},
        stopwords_to_remove={'down'},
        tokenizer=custom_tokenizer
    )._nlp
    assert isinstance(nlp, spacy.lang.en.English)
    # if we do nlp() rather than make_doc, we should still get lemmas/etc.
    doc = nlp(text)
    df = doc_to_dataframe(doc)
    assert not df.drop(columns=['ent_type_', 'ent_iob_']).replace('', np.nan).isna().any().any()
    custom_tokens = [str(x) for x in doc]
    assert custom_tokens == [
        'This', ':', '_is_', '_some_', '#text', 'down', 'dear', 'regards', '.'
    ]
    default_df = doc_to_dataframe(doc, include_punctuation=True)
    assert not default_df.query("text == 'down'").is_stop.iloc[0]
    assert default_df.query("text == 'dear'").is_stop.iloc[0]
    assert default_df.query("text == 'regards'").is_stop.iloc[0]
    assert (default_df['pos_'] == 'PUNCT').any()


def test__extract_lemmas(reddit):
    text = reddit['post'].iloc[2]
    doc = SpacyWrapper()._nlp(text)
    lemmas = extract_lemmas(doc)
    dataframe_to_text_file(
        pd.DataFrame(lemmas),
        get_test_file_path('text_preparation/extract_lemmas.txt')
    )


def test__extract_noun_phrases(reddit):
    text = reddit['post'].iloc[2]
    doc = SpacyWrapper()._nlp(text)
    phrases = extract_noun_phrases(doc)
    dataframe_to_text_file(
        pd.DataFrame(phrases),
        get_test_file_path('text_preparation/extract_noun_phrases.txt')
    )


def test__extract_named_entities(reddit):
    text = reddit['post'].iloc[2]
    doc = SpacyWrapper()._nlp(text)
    phrases = extract_named_entities(doc)
    dataframe_to_text_file(
        pd.DataFrame(phrases),
        get_test_file_path('text_preparation/extract_named_entities.txt')
    )


def test__extract_bi_grams(reddit):
    text = reddit['post'].iloc[2]
    doc = SpacyWrapper()._nlp(text)
    grams = extract_n_grams(doc)
    dataframe_to_text_file(
        pd.DataFrame(grams),
        get_test_file_path('text_preparation/extract_bi_grams.txt')
    )


def test__extract_from_doc(reddit):
    text = reddit['post'].iloc[2]
    doc = SpacyWrapper()._nlp(text)
    entities = extract_from_doc(doc)
    with open(get_test_file_path('text_preparation/extract_from_doc.txt'), 'w') as handle:
        for key, values in entities.items():
            handle.writelines(key + "\n")
            handle.writelines(str(values) + "\n")


def test__SpacyWrapper(reddit):
    stopwords_add = {
        'buy',
        'random',
        '_number_',
        '_url_',
    }
    stopwords_remove = {
        'down'
    }

    # ensure stopwords are A) not already stopwords, B) in the original documents, C) removed from
    # tokenized results
    spacy = SpacyWrapper()
    # first ensure that our custom stop words are not in the default list of stopwords (so that we
    # can later check that they are in the list of stopwords)
    assert spacy.stop_words.isdisjoint(stopwords_add)
    assert stopwords_remove < spacy.stop_words

    def _clean(x):
        x = clean(x)
        return x.replace('-', ' ')

    cleaned_posts = [_clean(x) for x in reddit['post']]
    results = spacy.extract(
        documents=cleaned_posts,
        all_lemmas=True,
        partial_lemmas=True,
        bi_grams=True,
        adjectives_verbs=True,
        nouns=True,
        noun_phrases=True,
        named_entities=True,
    )

    def _flatten(x: list[list]) -> set:
        return set([item for sublist in x for item in sublist])

    def _save_tokens(x: list[list], file_name: str):
        with open(get_test_file_path(f'text_preparation/{file_name}.txt'), 'w') as handle:
            for tokens in x:
                handle.writelines(' '.join(tokens) + "\n")

    all_lemmas = [x['all_lemmas'] for x in results]
    assert stopwords_add < _flatten(all_lemmas)  # ensure our extra stopwords appear
    assert stopwords_remove < _flatten(all_lemmas)  # ensure the stopwords that we want to remove appear in our dataset
    _save_tokens(all_lemmas, 'spacywrapper__all_lemmas')

    partial_lemmas = [x['partial_lemmas'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(partial_lemmas))
    # the stop words that we are going to remove should not appear (because they are currently stopwords)  # noqa
    assert stopwords_remove.isdisjoint(_flatten(partial_lemmas))
    _save_tokens(partial_lemmas, 'spacywrapper__partial_lemmas')

    bi_grams = _flatten([item.split('-') for sublist in results for item in sublist['bi_grams']])
    assert spacy.stop_words.isdisjoint(bi_grams)
    assert stopwords_remove.isdisjoint(bi_grams)
    _save_tokens(
        [x['bi_grams'] for x in results],
        'spacywrapper__bi_grams'
    )

    adjectives_verbs = [x['adjs_verbs'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(adjectives_verbs))
    _save_tokens(adjectives_verbs, 'spacywrapper__adjectives_verbs')

    nouns = [x['nouns'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(nouns))
    _save_tokens(nouns, 'spacywrapper__nouns')

    noun_phrases = _flatten([item.split('-') for sublist in results for item in sublist['noun_phrases']])  # noqa
    assert noun_phrases.isdisjoint(spacy.stop_words)
    assert stopwords_remove.isdisjoint(noun_phrases)
    _save_tokens(
        [x['noun_phrases'] for x in results],
        'spacywrapper__noun_phrases'
    )

    named_entities = [x['entities'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(named_entities))
    _save_tokens(named_entities, 'spacywrapper__named_entities')

    # test with additional stopwords
    spacy = SpacyWrapper(
        stopwords_to_add=stopwords_add,
        stopwords_to_remove=stopwords_remove,
    )
    # now our stopwords_add should be a subset of the stopwords
    assert stopwords_add < spacy.stop_words
    # and our stopwords_remove should not be a subset
    assert stopwords_remove.isdisjoint(spacy.stop_words)

    results = spacy.extract(
        documents=cleaned_posts,
        all_lemmas=True,
        partial_lemmas=True,
        bi_grams=True,
        adjectives_verbs=True,
        nouns=True,
        noun_phrases=True,
        named_entities=True,
    )

    all_lemmas = [x['all_lemmas'] for x in results]
    assert stopwords_add <= _flatten(all_lemmas)  # ensure our extra stopwords appear
    assert stopwords_remove < _flatten(all_lemmas)
    _save_tokens(all_lemmas, 'spacywrapper__all_lemmas__stopwords')

    partial_lemmas = [x['partial_lemmas'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(partial_lemmas))
    # now that we have removed these stopwords they should show up
    assert stopwords_remove < _flatten(partial_lemmas)
    _save_tokens(partial_lemmas, 'spacywrapper__partial_lemmas__stopwords')

    bi_grams = _flatten([item.split('-') for sublist in results for item in sublist['bi_grams']])
    assert spacy.stop_words.isdisjoint(bi_grams)
    # now that we have removed these stopwords they should show up
    assert stopwords_remove < bi_grams
    _save_tokens(
        [x['bi_grams'] for x in results],
        'spacywrapper__bi_grams__stopwords'
    )

    adjectives_verbs = [x['adjs_verbs'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(adjectives_verbs))
    assert stopwords_remove < _flatten(adjectives_verbs)
    _save_tokens(adjectives_verbs, 'spacywrapper__adjectives_verbs__stopwords')

    nouns = [x['nouns'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(nouns))
    _save_tokens(nouns, 'spacywrapper__nouns__stopwords')

    noun_phrases = _flatten([item.split('-') for sublist in results for item in sublist['noun_phrases']])  # noqa
    assert noun_phrases.isdisjoint(spacy.stop_words)
    assert stopwords_remove < noun_phrases
    _save_tokens(
        [x['noun_phrases'] for x in results],
        'spacywrapper__noun_phrases__stopwords'
    )

    named_entities = [x['entities'] for x in results]
    assert spacy.stop_words.isdisjoint(_flatten(named_entities))
    _save_tokens(named_entities, 'spacywrapper__named_entities__stopwords')


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
