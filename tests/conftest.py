import pytest
import pandas as pd
from source.library.datasets import CorpusDataLoader, CsvDataLoader, DatasetsBase, \
    PickledDataLoader, create_reddit_corpus_object, create_un_corpus_object

from tests.helpers import get_test_file_path
from source.library.spacy import Corpus
from source.library.text_preparation import clean


@pytest.fixture(scope='session')
def reddit():
    return pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))


@pytest.fixture(scope='function')
def documents_fake():
    return [
        '',
        '<b>This is 1 document that has very important information</b> and some $ & # characters.',
        'I',
        'This is another (the 2nd i.e. 2) unimportant document with almost no #awesome information!!',  # noqa
        '  ',
        "However, this is 'another' document with hyphen-word. & It has two sentences and is dumb."
    ]


@pytest.fixture(scope='function')
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


@pytest.fixture(scope='session')
def corpus_reddit(reddit):
    reddit = reddit.copy()
    # this Corpus is fit() without parallelization
    # it is used for base tests and to compare against Corpus fit() with parallelization
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
    # NOTE: specify
    corpus.fit(documents=reddit['post'].tolist(), num_batches=None)
    # make sure these didn't reset during fitting
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    return corpus


@pytest.fixture(scope='session')
def corpus_reddit_parallel(reddit):
    reddit = reddit.copy()
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


class TestDatasets(DatasetsBase):
    def __init__(self) -> None:
        # define the datasets before calling __init__()
        self.dataset_1 = PickledDataLoader(
            description="Dataset description",
            dependencies=['SNOWFLAKE.SCHEMA.TABLE'],
            directory='.',
            cache=True,
        )
        self.other_dataset_2 = PickledDataLoader(
            description="Other dataset description",
            dependencies=['dataset_1'],
            directory='.',
            cache=True,
        )
        self.dataset_3_csv = CsvDataLoader(
            description="Other dataset description",
            dependencies=['other_dataset_2'],
            directory='.',
            cache=True,
        )
        super().__init__()


class TestCorpusDatasets(DatasetsBase):
    def __init__(self, cache: bool) -> None:
        # define the datasets before calling __init__()
        self.corpus_reddit = CorpusDataLoader(
            description="Reddit Corpus description",
            dependencies=['other_dataset_1'],
            directory='.',
            corpus_creator=create_reddit_corpus_object,
            batch_size=100,
            cache=cache,
        )
        self.corpus_un = CorpusDataLoader(
            description="UN Corpus description",
            dependencies=['other_dataset_2'],
            directory='.',
            corpus_creator=create_un_corpus_object,
            cache=cache,
        )
        super().__init__()


@pytest.fixture(scope='function')
def datasets_fake():
    return TestDatasets()


@pytest.fixture(scope='function')
def datasets_corpus_fake():
    return TestCorpusDatasets(cache=True)
