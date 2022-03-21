"""
This code is from:

    Blueprints for Text Analytics Using Python
    by Jens Albrecht, Sidharth Ramachandran, and Christian Winkler
    (O'Reilly, 2021), 978-1-492-07408-3.
    https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
"""
from spacy.lang.en import English

excluded_stop_words = {'down'}
included_stop_words = {'dear', 'regards'}


class CustomEnglishDefaults(English.Defaults):
    stop_words = English.Defaults.stop_words.copy()
    stop_words -= excluded_stop_words
    stop_words |= included_stop_words


class CustomEnglish(English):
    Defaults = CustomEnglishDefaults
