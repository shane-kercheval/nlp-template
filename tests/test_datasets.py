import os
import shutil
import pandas as pd
import numpy as np
from source.library.datasets import create_reddit_corpus_object
from tests.conftest import TestCorpusDatasets


def test__datasets(datasets_fake):
    data = datasets_fake
    assert data.datasets == ['dataset_1', 'dataset_3_csv', 'other_dataset_2']
    assert data.descriptions == [
        {'dataset': 'dataset_1', 'description': 'Dataset description'},
        {'dataset': 'dataset_3_csv', 'description': 'Other dataset description'},
        {'dataset': 'other_dataset_2', 'description': 'Other dataset description'},
    ]
    assert data.dependencies == [
        {'dataset': 'dataset_1', 'dependencies': ['SNOWFLAKE.SCHEMA.TABLE']},
        {'dataset': 'dataset_3_csv', 'dependencies': ['other_dataset_2']},
        {'dataset': 'other_dataset_2', 'dependencies': ['dataset_1']},
    ]
    assert data.dataset_1._cached_data is None
    assert data.dataset_1.path == './dataset_1.pkl'
    assert not os.path.isfile(data.dataset_1.path)
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    data.dataset_1.save(df)
    assert data.dataset_1._cached_data is not None
    assert os.path.isfile(data.dataset_1.path)
    data.dataset_1.clear_cache()
    assert data.dataset_1._cached_data is None
    df_loaded = data.dataset_1.load()
    assert data.dataset_1._cached_data is not None
    assert (df == df_loaded).all().all()

    assert data.other_dataset_2._cached_data is None
    assert data.other_dataset_2.path == './other_dataset_2.pkl'
    assert not os.path.isfile(data.other_dataset_2.path)
    data.other_dataset_2.save(df.replace(1, 10))
    assert data.other_dataset_2._cached_data is not None
    assert os.path.isfile(data.other_dataset_2.path)
    data.other_dataset_2.clear_cache()
    assert data.other_dataset_2._cached_data is None
    df_loaded = data.other_dataset_2.load()
    assert data.other_dataset_2._cached_data is not None
    assert (df.replace(1, 10) == df_loaded).all().all()

    assert data.dataset_3_csv._cached_data is None
    assert data.dataset_3_csv.path == './dataset_3_csv.csv'
    assert not os.path.isfile(data.dataset_3_csv.path)
    data.dataset_3_csv.save(df.replace(1, 10))
    assert data.dataset_3_csv._cached_data is not None
    assert os.path.isfile(data.dataset_3_csv.path)
    data.dataset_3_csv.clear_cache()
    assert data.dataset_3_csv._cached_data is None
    df_loaded = data.dataset_3_csv.load()
    assert data.dataset_3_csv._cached_data is not None
    assert (df.replace(1, 10) == df_loaded).all().all()

    os.remove('./dataset_1.pkl')
    os.remove('./other_dataset_2.pkl')
    os.remove('./dataset_3_csv.csv')


def _assert_tokens_equal(loaded_token, original_token):
    """Tests that a token that was loaded from json/dict matches the original."""
    assert loaded_token.text is not None
    assert loaded_token.text == original_token.text
    assert loaded_token.lemma is not None
    assert loaded_token.lemma == original_token.lemma
    assert loaded_token.is_stop_word is not None
    assert loaded_token.is_stop_word == original_token.is_stop_word
    assert loaded_token.embeddings is not None
    assert isinstance(loaded_token.embeddings, np.ndarray)
    assert (loaded_token.embeddings == original_token.embeddings).all()
    assert loaded_token.part_of_speech is not None
    assert loaded_token.part_of_speech == original_token.part_of_speech
    assert loaded_token.is_punctuation is not None
    assert loaded_token.is_punctuation == original_token.is_punctuation
    assert loaded_token.is_special is not None
    assert loaded_token.is_special == original_token.is_special
    assert loaded_token.is_alpha is not None
    assert loaded_token.is_alpha == original_token.is_alpha
    assert loaded_token.is_numeric is not None
    assert loaded_token.is_numeric == original_token.is_numeric
    assert loaded_token.is_ascii is not None
    assert loaded_token.is_ascii == original_token.is_ascii
    assert loaded_token.dep is not None
    assert loaded_token.dep == original_token.dep
    assert loaded_token.entity_type is not None
    assert loaded_token.entity_type == original_token.entity_type
    assert loaded_token.sentiment is not None
    assert loaded_token.sentiment == original_token.sentiment


def test__datasets__corpus(datasets_corpus_fake, corpus_simple_example):
    data = datasets_corpus_fake
    assert data.datasets == ['corpus_reddit', 'corpus_un']
    assert data.descriptions == [
        {'dataset': 'corpus_reddit', 'description': 'Reddit Corpus description'},
        {'dataset': 'corpus_un', 'description': 'UN Corpus description'},
    ]
    assert data.dependencies == [
        {'dataset': 'corpus_reddit', 'dependencies': ['other_dataset_1']},
        {'dataset': 'corpus_un', 'dependencies': ['other_dataset_2']},
    ]
    assert data.corpus_reddit._cached_data is None

    assert data.corpus_reddit.sub_directory == './corpus_reddit__json'
    assert not os.path.exists(data.corpus_reddit.sub_directory)
    data.corpus_reddit.save(corpus_simple_example)
    assert os.path.exists(data.corpus_reddit.sub_directory)
    assert data.corpus_reddit._cached_data is not None
    # loading the cached value
    loaded_corpus = data.corpus_reddit.load()
    # for each doc test all attributes of document are the same and retest all tokens match
    for original_doc, loaded_doc in zip(corpus_simple_example, loaded_corpus):
        assert original_doc._text_original == loaded_doc._text_original
        assert original_doc._text_cleaned == loaded_doc._text_cleaned
        for loaded_t, original_t in zip(loaded_doc._tokens, original_doc._tokens):
            _assert_tokens_equal(loaded_t, original_t)
    # clear cache and re-load
    data.corpus_reddit.clear_cache()
    assert data.corpus_reddit._cached_data is None
    loaded_corpus = data.corpus_reddit.load()
    shutil.rmtree(data.corpus_reddit.sub_directory)
    # for each doc test all attributes of document are the same and retest all tokens match
    for original_doc, loaded_doc in zip(corpus_simple_example, loaded_corpus):
        assert original_doc._text_original == loaded_doc._text_original
        assert original_doc._text_cleaned == loaded_doc._text_cleaned
        for loaded_t, original_t in zip(loaded_doc._tokens, original_doc._tokens):
            _assert_tokens_equal(loaded_t, original_t)


def test__datasets__corpus__large_files_ensure_same_order(
        datasets_corpus_fake: TestCorpusDatasets,
        reddit: pd.DataFrame):
    reddit = reddit.copy()
    # We need to the documents are in the same order when they are read back in;
    # we can't rely on how files are sorted because it might be sorted like this:
    # file_1
    # file_10
    # file_11
    # ...
    # file_2
    # file_20
    data = datasets_corpus_fake
    reddit_corpus = create_reddit_corpus_object()
    reddit_corpus.fit(documents=reddit['post'].tolist())
    # save data to file and reload, then test that the re-loaded corpus has the same functionality
    # as the original
    data.corpus_reddit.save(reddit_corpus)
    assert data.corpus_reddit._cached_data is not None
    data.corpus_reddit.clear_cache()
    assert data.corpus_reddit._cached_data is None
    loaded_corpus = data.corpus_reddit.load()
    shutil.rmtree(data.corpus_reddit.sub_directory)

    # for each doc test all attributes of document are the same and test that all tokens match
    for original_doc, new_doc in zip(reddit_corpus, loaded_corpus):
        assert original_doc._text_original == new_doc._text_original
        assert original_doc._text_cleaned == new_doc._text_cleaned
        assert (original_doc.token_embeddings() == new_doc.token_embeddings()).all()
        assert (original_doc.embeddings() == new_doc.embeddings()).all()
        for loaded_t, original_t in zip(new_doc._tokens, original_doc._tokens):
            _assert_tokens_equal(loaded_t, original_t)

    ####
    # Lets make sure the vectorizers/etc. produce the same results
    ####
    test_text_value = "This is a post about BMW cars and how to fix my engine"
    orig_vector = reddit_corpus.\
        text_to_count_vector(test_text_value).\
        toarray()
    assert orig_vector.sum() > 1
    new_vector = loaded_corpus.\
        text_to_count_vector(test_text_value).\
        toarray()
    assert (orig_vector == new_vector).all()

    orig_vector = reddit_corpus.\
        text_to_tf_idf_vector(test_text_value).\
        toarray()
    assert orig_vector.sum() > 0
    new_vector = loaded_corpus.\
        text_to_tf_idf_vector(test_text_value).\
        toarray()
    assert (orig_vector == new_vector).all()

    assert (reddit_corpus.embeddings_matrix() == loaded_corpus.embeddings_matrix()).all()
    assert (reddit_corpus.count_vectorizer_vocab() == loaded_corpus.count_vectorizer_vocab()).all()  # noqa
    assert (reddit_corpus.tf_idf_vectorizer_vocab() == loaded_corpus.tf_idf_vectorizer_vocab()).all()  # noqa
    assert (reddit_corpus.similarity_matrix(how='embeddings-tf_idf') == loaded_corpus.similarity_matrix(how='embeddings-tf_idf')).all()  # noqa
    assert (reddit_corpus.calculate_similarities(text=test_text_value, how='embeddings-tf_idf') == loaded_corpus.calculate_similarities(text=test_text_value, how='embeddings-tf_idf')).all()  # noqa
