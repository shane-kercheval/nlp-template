import unittest

import pandas as pd

from source.library.text_preparation import clean, doc_to_dataframe, custom_tokenizer, extract_lemmas, extract_noun_phrases, extract_named_entities, \
    extract_nlp
from source.tests.helpers import get_test_file_path, dataframe_to_text_file
from source.library.text_cleaning_simple import tokenize


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


    def test__asdf(self):
        import spacy

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

        extract_nlp(doc)



