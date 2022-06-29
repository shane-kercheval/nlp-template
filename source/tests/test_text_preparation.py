import unittest

import fasttext
import numpy as np
import pandas as pd
import spacy
# from spacy.lang.en import English

from source.library.spacy import doc_to_dataframe, custom_tokenizer, extract_lemmas, extract_noun_phrases, \
    extract_named_entities, create_spacy_pipeline, extract_from_doc, get_stopwords, extract_n_grams
from source.library.text_preparation import clean, predict_language
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


class TestTextPreparation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reddit = pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))
        cls.reddit = reddit

    def test__clean(self):
        text = "<This> 😩😬 [sentence] [sentence]& a & [link to something](www.hotmail.com) stuff && abc --" \
            "- -   https://www.google.com/search?q=asdfa; john.doe@gmail.com #remove 445 583.345.7833 @shane"
        clean_text = clean(text)
        expected_text = '_EMOJI_ _EMOJI_ a link to something stuff abc - _URL_ _EMAIL_ _TAG_ _NUMBER_ ' \
            '_PHONE_ _USER_'
        self.assertEqual(expected_text, clean_text)

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
        self.assertEqual(expected_text, clean_text)

        with open(get_test_file_path('text_preparation/clean__reddit.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in self.reddit['post'].apply(clean)])

        with open(get_test_file_path('text_preparation/example_unclean.txt'), 'r') as handle:
            text_lines = handle.readlines()

        with open(get_test_file_path('text_preparation/example_clean.txt'), 'w') as handle:
            handle.writelines([clean(x) + "\n" for x in text_lines])

    def test__get_stopwords(self):
        stopwords = get_stopwords()
        self.assertTrue(len(stopwords) > 300)

    def test__create_spacy_pipeline(self):
        text = "This: _is_ _some_ #text down dear regards."

        nlp = create_spacy_pipeline()
        self.assertIsInstance(nlp, spacy.lang.en.English)
        # if we do nlp() rather than make_doc, we should still get lemmas/etc.
        doc = nlp(text)
        df = doc_to_dataframe(doc)
        self.assertFalse(df.drop(columns=['ent_type_', 'ent_iob_']).replace('', np.nan).isna().any().any())
        default_tokens = [str(x) for x in doc]
        self.assertEqual(
            default_tokens,
            ['This', ':', '_', 'is', '_', '_', 'some', '_', '#', 'text', 'down', 'dear', 'regards', '.']
        )
        default_df = doc_to_dataframe(doc, include_punctuation=True)
        self.assertTrue(default_df.query("text == 'down'").is_stop.iloc[0])
        self.assertFalse(default_df.query("text == 'dear'").is_stop.iloc[0])
        self.assertFalse(default_df.query("text == 'regards'").is_stop.iloc[0])
        self.assertTrue((default_df['pos_'] == 'PUNCT').any())
        self.assertFalse((doc_to_dataframe(doc, include_punctuation=False)['pos_'] == 'PUNCT').any())

        nlp = create_spacy_pipeline(
            stopwords_to_add={'dear', 'regards'},
            stopwords_to_remove={'down'},
            tokenizer=custom_tokenizer
        )
        self.assertIsInstance(nlp, spacy.lang.en.English)
        # if we do nlp() rather than make_doc, we should still get lemmas/etc.
        doc = nlp(text)
        df = doc_to_dataframe(doc)
        self.assertFalse(df.drop(columns=['ent_type_', 'ent_iob_']).replace('', np.nan).isna().any().any())
        custom_tokens = [str(x) for x in doc]
        self.assertEqual(
            custom_tokens,
            ['This', ':', '_is_', '_some_', '#text', 'down', 'dear', 'regards', '.']
        )
        default_df = doc_to_dataframe(doc, include_punctuation=True)
        self.assertFalse(default_df.query("text == 'down'").is_stop.iloc[0])
        self.assertTrue(default_df.query("text == 'dear'").is_stop.iloc[0])
        self.assertTrue(default_df.query("text == 'regards'").is_stop.iloc[0])
        self.assertTrue((default_df['pos_'] == 'PUNCT').any())

    def test_extract_lemmas(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        lemmas = extract_lemmas(doc)
        dataframe_to_text_file(
            pd.DataFrame(lemmas),
            get_test_file_path('text_preparation/extract_lemmas.txt')
        )

    def test_extract_noun_phrases(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        phrases = extract_noun_phrases(doc)
        dataframe_to_text_file(
            pd.DataFrame(phrases),
            get_test_file_path('text_preparation/extract_noun_phrases.txt')
        )

    def test_extract_named_entities(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        phrases = extract_named_entities(doc)
        dataframe_to_text_file(
            pd.DataFrame(phrases),
            get_test_file_path('text_preparation/extract_named_entities.txt')
        )

    def test_extract_bi_grams(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        grams = extract_n_grams(doc)
        dataframe_to_text_file(
            pd.DataFrame(grams),
            get_test_file_path('text_preparation/extract_bi_grams.txt')
        )

    def test_extract_from_doc(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        entities = extract_from_doc(doc)
        with open(get_test_file_path('text_preparation/extract_from_doc.txt'), 'w') as handle:
            for key, values in entities.items():
                handle.writelines(key + "\n")
                handle.writelines(str(values) + "\n")

    def test_predict_language(self):
        model = fasttext.load_model("/fasttext/lid.176.ftz")
        self.assertEqual(
            predict_language('This is english.', model=model, return_language_code=False),
            "English"
        )
        self.assertEqual(
            predict_language('This is english.', model=model, return_language_code=True),
            "en"
        )

        language_text = [
            "I don't like version 2.0 of Chat4you 😡👎",  # English
            "Ich mag Version 2.0 von Chat4you nicht 😡👎",  # German
            "Мне не нравится версия 2.0 Chat4you 😡👎",  # Russian
            "Não gosto da versão 2.0 do Chat4you 😡👎",  # Portuguese
            "मुझे Chat4you का संस्करण 2.0 पसंद नहीं है 😡👎"]  # Hindi
        language_text_df = pd.Series(language_text, name='text').to_frame()

        # create new column
        language_text_df['language'] = language_text_df['text'].apply(predict_language, model=model)
        language_text_df['language_code'] = language_text_df['text'].apply(
            predict_language,
            model=model,
            return_language_code=True
        )
        dataframe_to_text_file(
            language_text_df,
            get_test_file_path('text_preparation/predict_language.txt')
        )
