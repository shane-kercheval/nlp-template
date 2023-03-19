import pytest
import pandas as pd

from tests.helpers import get_test_file_path
from source.library.spacy import Corpus
from source.library.text_preparation import clean


@pytest.fixture
def reddit():
    return pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))


@pytest.fixture
def documents_fake():
    return [
        '',
        '<b>This is 1 document that has very important information</b> and some $ & # characters.',
        'I',
        'This is another (the 2nd i.e. 2) unimportant document with almost no #awesome information!!',  # noqa
        '  ',
        "However, this is 'another' document with hyphen-word. & It has two sentences and is dumb."
    ]


@pytest.fixture
def corpus_simple_example(documents_fake):
    stop_words_to_add = {'dear', 'regard'}
    stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}

    corpus = Corpus()
    assert all(x not in corpus.stop_words for x in stop_words_to_add)
    assert all(x in corpus.stop_words for x in stop_words_to_remove)

    corpus = Corpus(
        stop_words_to_add=stop_words_to_add,
        stop_words_to_remove=stop_words_to_remove,
        pre_process=clean,
        spacy_model='en_core_web_md',
        sklearn_tokenenizer_min_df=1,
    )
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    corpus.fit(documents=documents_fake)
    # make sure these didn't reset during fitting
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    return corpus


@pytest.fixture
def corpus_reddit(reddit):
    stop_words_to_add = {'dear', 'regard', '_number_', '_tag_'}
    stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}

    corpus = Corpus()
    assert all(x not in corpus.stop_words for x in stop_words_to_add)
    assert all(x in corpus.stop_words for x in stop_words_to_remove)

    corpus = Corpus(
        stop_words_to_add=stop_words_to_add,
        stop_words_to_remove=stop_words_to_remove,
        pre_process=clean,
        spacy_model='en_core_web_sm',
        sklearn_tokenenizer_min_df=1,
    )
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    corpus.fit(documents=reddit['post'].tolist())
    # make sure these didn't reset during fitting
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    return corpus
