import unittest

import pandas as pd
import spacy
from spacy.lang.en import English
from helpsk.utility import redirect_stdout_to_file

from source.library.space_language import CustomEnglish
from source.library.spacy import doc_to_dataframe, custom_tokenizer, extract_lemmas, extract_noun_phrases, \
    extract_named_entities, create_spacy_pipeline, extract_from_doc
from source.library.text_preparation import clean, predict_language
from source.tests.helpers import get_test_file_path, dataframe_to_text_file


class TestTextPreparation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reddit = pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))
        cls.reddit = reddit

    def test__clean(self):
        text = "<This> 😩😬 [sentence] [sentence]& a & [link to something](www.hotmail.com) stuff && abc --- -   https://www.google.com/search?q=asdfa; john.doe@gmail.com #remove 445 583.345.7833 @shane"  # noqa
        clean_text = clean(text)
        expected_text = '_EMOJI_ _EMOJI_ a link to something stuff abc - _URL_ _EMAIL_ _TAG_ _NUMBER_ _PHONE_ _USER_'
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
        expected_text = 'This 😩😬 sentence sentence a link to something stuff abc - https://www.google.com/search?q=asdfa; _EMAIL_ #remove 445 _PHONE_ @shane'  # noqa
        self.assertEqual(expected_text, clean_text)

        with open(get_test_file_path('text_preparation/clean__reddit.txt'), 'w') as handle:
            handle.writelines([x + "\n" for x in self.reddit['post'].apply(clean)])

        with open(get_test_file_path('text_preparation/example_unclean.txt'), 'r') as handle:
            text_lines = handle.readlines()

        with open(get_test_file_path('text_preparation/example_clean.txt'), 'w') as handle:
            handle.writelines([clean(x) + "\n" for x in text_lines])

    def test__custom_english(self):
        language_default = English()
        stop_words_default = language_default.Defaults.stop_words
        self.assertTrue('down' in stop_words_default)
        language_custom = CustomEnglish()
        stop_words_custom = language_custom.Defaults.stop_words

        self.assertEqual(stop_words_custom - stop_words_default,
                         {'dear', 'regards'})
        self.assertEqual(stop_words_default - stop_words_custom,
                         {'down'})

        doc = language_custom.make_doc("Hey dear I'd like to go down town. Regards")
        doc_df = doc_to_dataframe(doc)
        self.assertFalse(doc_df.query("text == 'down'").is_stop.iloc[0])
        self.assertTrue(doc_df.query("text == 'dear'").is_stop.iloc[0])
        self.assertTrue(doc_df.query("text == 'Regards'").is_stop.iloc[0])

    def test__create_spacy_pipeline(self):
        text = "This: _is_ _some_ #text down dear regards."
        nlp = create_spacy_pipeline()
        self.assertIsInstance(nlp, spacy.lang.en.English)
        doc = nlp.make_doc(text)
        default_tokens = [str(x) for x in doc]
        self.assertEqual(default_tokens,
                         ['This', ':', '_', 'is', '_', '_', 'some', '_', '#', 'text', 'down', 'dear', 'regards', '.'])
        default_df = doc_to_dataframe(doc, include_punctuation=True)
        self.assertTrue(default_df.query("text == 'down'").is_stop.iloc[0])
        self.assertFalse(default_df.query("text == 'dear'").is_stop.iloc[0])
        self.assertFalse(default_df.query("text == 'regards'").is_stop.iloc[0])

        nlp = create_spacy_pipeline(language=CustomEnglish,
                                    tokenizer=custom_tokenizer)
        self.assertIsInstance(nlp, CustomEnglish)
        doc = nlp.make_doc(text)
        custom_tokens = [str(x) for x in doc]
        self.assertEqual(custom_tokens,
                         ['This', ':', '_is_', '_some_', '#text', 'down', 'dear', 'regards', '.'])
        default_df = doc_to_dataframe(doc, include_punctuation=True)
        self.assertFalse(default_df.query("text == 'down'").is_stop.iloc[0])
        self.assertTrue(default_df.query("text == 'dear'").is_stop.iloc[0])
        self.assertTrue(default_df.query("text == 'regards'").is_stop.iloc[0])

    def test_extract_lemmas(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        lemmas = extract_lemmas(doc)
        dataframe_to_text_file(pd.DataFrame(lemmas),
                               get_test_file_path('text_preparation/extract_lemmas.txt'))

    def test_extract_noun_phrases(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        phrases = extract_noun_phrases(doc)
        dataframe_to_text_file(pd.DataFrame(phrases),
                               get_test_file_path('text_preparation/extract_noun_phrases.txt'))

    def test_extract_named_entities(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        phrases = extract_named_entities(doc)
        dataframe_to_text_file(pd.DataFrame(phrases),
                               get_test_file_path('text_preparation/extract_named_entities.txt'))

    def test_extract_from_doc(self):
        text = self.reddit['post'].iloc[2]
        doc = create_spacy_pipeline()(text)
        entities = extract_from_doc(doc)
        with open(get_test_file_path('text_preparation/extract_from_doc.txt'), 'w') as handle:
            for key, values in entities.items():
                handle.writelines(key + "\n")
                handle.writelines(str(values) + "\n")




    def test__asdf(self):

        text = self.reddit['post'].iloc[2]
        text = clean(text=text)





        # DEFAULT TOKENIZER
        import spacy
        nlp = spacy.load("en_core_web_sm")
        type(nlp)
        doc = nlp(text)
        type(doc)

        for token in doc:
            print(token, end="|")


        # CUSTOM TOKENIZER
        nlp = spacy.load('en_core_web_sm')
        nlp.tokenizer = custom_tokenizer(nlp)

        doc = nlp(text)
        for token in doc:
            print(token, end="|")

        doc_to_dataframe(doc)

        temp = doc_to_dataframe(doc)
        lemmas = extract_lemmas(doc, include_part_of_speech=None)
        for lemma in lemmas:
            print(lemma, end="|")

        lemmas = extract_lemmas(doc, include_part_of_speech=['ADJ', 'NOUN'])
        for lemma in lemmas:
            print(lemma, end="|")

        phrases = extract_noun_phrases(doc)
        for phrase in phrases:
            print(phrase, end="|")

        ents = extract_named_entities(doc)
        for ent in ents:
            print(ent, end="|")

        extract_from_doc(doc)

        import fasttext

        import os
        os.getcwd()
        lang_model = fasttext.load_model("../../lid.176.ftz")


        # make a prediction
        print(lang_model.predict('Good morning', 3))


        predict_language(text)

        langauge_text = [
            "I don't like version 2.0 of Chat4you 😡👎",  # English
            "Ich mag Version 2.0 von Chat4you nicht 😡👎",  # German
            "Мне не нравится версия 2.0 Chat4you 😡👎",  # Russian
            "Não gosto da versão 2.0 do Chat4you 😡👎",  # Portugese
            "मुझे Chat4you का संस्करण 2.0 पसंद नहीं है 😡👎"]  # Hindi
        langauge_text_df = pd.Series(langauge_text, name='text').to_frame()

        # create new column
        langauge_text_df['lang'] = langauge_text_df['text'].apply(predict_language)
        langauge_text_df

        lang_df = pd.read_csv('../resources/language_codes.csv')
        lang_df = lang_df[['name', '639-1', '639-2']].melt(id_vars=['name'], var_name='iso', value_name='code')
        iso639_languages = lang_df.set_index('code')['name'].to_dict()

        langauge_text_df['lang_name'] = langauge_text_df['lang'].map(iso639_languages)
        langauge_text_df


        # Not in book: Normalize tokens with a dict
        token_map = {'U.S.': 'United_States',
                     'L.A.': 'Los_Angeles'}

        def token_normalizer(tokens):
            return [token_map.get(t, t) for t in tokens]

        tokens = "L.A. is a city in the U.S.".split()
        tokens = token_normalizer(tokens)

        print(*tokens, sep='|')



class Temp():
    temp = 1

