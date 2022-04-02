import unittest

import fasttext
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en import English

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
        text = "<This> 😩😬 [sentence] [sentence]& a & [link to something](www.hotmail.com) stuff && abc --- -   https://www.google.com/search?q=asdfa; john.doe@gmail.com #remove 445 583.345.7833 @shane"  # noqa
        clean_text = clean(text)
        expected_text = '_EMOJI_ _EMOJI_ a link to something stuff abc - _URL_ _EMAIL_ _TAG_ _NUMBER_ _PHONE_ _USER_'  # noqa
        self.assertEqual(expected_text, clean_text)
