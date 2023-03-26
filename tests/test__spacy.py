import json
import pandas as pd
import numpy as np
import spacy as sp

from source.library.spacy import Corpus, Document, Token, STOP_WORDS_DEFAULT
from source.library.text_preparation import clean

from tests.helpers import dataframe_to_text_file, get_test_file_path


def test__corpus__empty_string(corpus_simple_example, documents_fake):
    corpus = corpus_simple_example
    assert len(corpus) == len(documents_fake)
    # first let's check the documents with edge cases
    assert corpus[0].text(original=True) == ''
    assert corpus[0].text(original=False) == ''
    assert corpus[2].text(original=True) == 'I'
    assert corpus[2].text(original=False) == 'I'
    assert corpus[4].text(original=True) == '  '
    assert corpus[4].text(original=False) == ''

    assert corpus[0].num_tokens(important_only=True) == 0
    assert corpus[2].num_tokens(important_only=True) == 0
    assert corpus[4].num_tokens(important_only=True) == 0

    assert list(corpus[0].lemmas(important_only=True)) == []
    assert list(corpus[0].lemmas(important_only=False)) == []
    assert list(corpus[2].lemmas(important_only=True)) == []
    assert list(corpus[2].lemmas(important_only=False)) == ['i']
    assert list(corpus[4].lemmas(important_only=True)) == []
    assert list(corpus[4].lemmas(important_only=False)) == []


def test__corpus__document(corpus_simple_example, documents_fake):
    corpus = corpus_simple_example
    non_empty_corpi = [corpus[1], corpus[3], corpus[5]]

    with open(get_test_file_path(f'spacy/document_processor__text.txt'), 'w') as handle:  # noqa
        handle.writelines([d.text(original=False) + "\n" for d in corpus])

    for index in range(len(documents_fake)):
        assert [t.text for t in corpus[index]] == [t.text for t in corpus[index]._tokens]
        assert len(corpus[index]) == len(corpus[index]._tokens)
        dataframe_to_text_file(
            pd.DataFrame(corpus[index].to_dictionary()),
            get_test_file_path(f'spacy/document_to_dictionary__sample_{index}.txt')
        )

    assert all(len(list(d.lemmas(important_only=True))) > 0 for d in non_empty_corpi)
    assert all(len(list(d.lemmas(important_only=False))) > 0 for d in non_empty_corpi)
    assert all(len(list(d.n_grams())) > 0 for d in non_empty_corpi)
    assert all(len(list(d.nouns())) > 0 for d in non_empty_corpi)
    assert all(len(list(d.noun_phrases())) > 0 for d in non_empty_corpi)
    assert all(len(list(d.adjectives_verbs())) > 0 for d in non_empty_corpi)
    assert all(len(list(d.entities())) > 0 for d in non_empty_corpi)
    assert [x.impurity() for x in corpus]  # assert it works and doesn't fail with empty docs
    assert all([x.sentiment() == 0 for x in corpus])  # not working yet

    # sanity check cache (e.g. which could fail with generators)
    assert corpus[1].impurity(original=False) == corpus[1].impurity(original=False)
    assert corpus[1].impurity(original=True) == corpus[1].impurity(original=True)
    assert all(d.impurity(original=True) > 0 for d in non_empty_corpi)
    assert all(d.impurity(original=True) > d.impurity(original=False) for d in non_empty_corpi)

    assert corpus[0].diff(use_lemmas=False)
    assert corpus[0].diff(use_lemmas=True)
    assert corpus[2].diff(use_lemmas=False)
    assert corpus[2].diff(use_lemmas=True)
    assert corpus[4].diff(use_lemmas=False)
    assert corpus[4].diff(use_lemmas=True)
    # sanity check cache (e.g. which could fail with generators)
    assert corpus[1].diff(use_lemmas=False) == corpus[1].diff(use_lemmas=False)
    assert corpus[1].diff(use_lemmas=True) == corpus[1].diff(use_lemmas=True)
    assert len(corpus[1].diff(use_lemmas=False)) > 0
    assert len(corpus[1].diff(use_lemmas=True)) > 0


def test__corpus__tokens(corpus_simple_example):
    corpus = corpus_simple_example
    with open(get_test_file_path('spacy/corpus__text__original__sample.txt'), 'w') as handle:
        handle.writelines(t + "\n" for t in corpus.text())

    with open(get_test_file_path('spacy/corpus__text__clean__sample.txt'), 'w') as handle:
        handle.writelines(t + "\n" for t in corpus.text(original=False))

    with open(get_test_file_path('spacy/corpus__lemmas__important_only__sample.txt'), 'w') as handle:  # noqa
        handle.writelines('|'.join(x) + "\n" for x in corpus.lemmas())

    with open(get_test_file_path('spacy/corpus__lemmas__important_only__false__sample.txt'), 'w') as handle:  # noqa
        handle.writelines('|'.join(x) + "\n" for x in corpus.lemmas(important_only=False))

    with open(get_test_file_path('spacy/corpus__n_grams__2__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.n_grams(2))

    with open(get_test_file_path('spacy/corpus__n_grams__3__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.n_grams(3, separator='--'))

    with open(get_test_file_path('spacy/corpus__nouns__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.nouns())

    with open(get_test_file_path('spacy/corpus__noun_phrases__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.noun_phrases())

    with open(get_test_file_path('spacy/corpus__adjectives_verbs__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.adjectives_verbs())

    with open(get_test_file_path('spacy/corpus__entities__sample.txt'), 'w') as handle:
        handle.writelines('|'.join(f"{e[0]} ({e[1]})" for e in x) + "\n" for x in corpus.entities())  # noqa

    with open(get_test_file_path('spacy/document_diff__empty.html'), 'w') as file:
        file.write(corpus[0].diff())

    with open(get_test_file_path('spacy/document_diff__empty__use_lemmas.html'), 'w') as file:
        file.write(corpus[0].diff(use_lemmas=True))


def _assert_tokens_equal(loaded_t: Token, original_t: Token):
    """Tests that a token that was loaded from json/dict matches the original."""
    assert loaded_t.text is not None
    assert loaded_t.text == original_t.text
    assert loaded_t.lemma is not None
    assert loaded_t.lemma == original_t.lemma
    assert loaded_t.is_stop_word is not None
    assert loaded_t.is_stop_word == original_t.is_stop_word
    assert loaded_t.embeddings is not None
    assert isinstance(loaded_t.embeddings, np.ndarray)
    assert (loaded_t.embeddings == original_t.embeddings).all()
    assert loaded_t.part_of_speech is not None
    assert loaded_t.part_of_speech == original_t.part_of_speech
    assert loaded_t.is_punctuation is not None
    assert loaded_t.is_punctuation == original_t.is_punctuation
    assert loaded_t.is_special is not None
    assert loaded_t.is_special == original_t.is_special
    assert loaded_t.is_alpha is not None
    assert loaded_t.is_alpha == original_t.is_alpha
    assert loaded_t.is_numeric is not None
    assert loaded_t.is_numeric == original_t.is_numeric
    assert loaded_t.is_ascii is not None
    assert loaded_t.is_ascii == original_t.is_ascii
    assert loaded_t.dep is not None
    assert loaded_t.dep == original_t.dep
    assert loaded_t.entity_type is not None
    assert loaded_t.entity_type == original_t.entity_type
    assert loaded_t.sentiment is not None
    assert loaded_t.sentiment == original_t.sentiment


def test__tokens__to_dict():
    text = "This is a single document with some text and 1 2 3 numbers and && # % symbols."
    nlp = sp.load('en_core_web_sm')
    tokens = nlp(text)
    # for each token, A) make sure we can serialize to json (e.g. that embeddings is converted
    # from a numpy array to a list) B) make sure we can load back in from json and C) make sure
    # the objects/values are unchanged
    for index, spacy_token in enumerate(tokens):
        parsed_token = Token.from_spacy(token=spacy_token, stop_words=STOP_WORDS_DEFAULT)
        _file_name = get_test_file_path(f'spacy/token__to_dict__index_{index}.txt')
        with open(_file_name, 'w') as f:
            json.dump(parsed_token.to_dict(), f)
        assert isinstance(parsed_token.embeddings, np.ndarray)
        with open(_file_name, 'r') as f:
            loaded_json = json.load(f)
        assert loaded_json
        loaded_token = Token.from_dict(loaded_json)
        _assert_tokens_equal(loaded_token, parsed_token)


def test__document__to_dict():
    text = "This is a single document with some text and 1 2 3 numbers and && # % symbols."
    text_clean = "This is a single document with some text and _number_ _number_ _number_ numbers and _symbol_ _symbol_ symbols."  # noqa
    nlp = sp.load('en_core_web_sm')
    tokens = nlp(text)

    original_doc = Document(
        tokens=[Token.from_spacy(token=t, stop_words=STOP_WORDS_DEFAULT) for t in tokens],
        text_original=text,
        text_cleaned=text_clean,
    )
    _file_name = get_test_file_path('spacy/document__to_dict__simple_example.txt')
    with open(_file_name, 'w') as f:
        json.dump(original_doc.to_dict(), f)
    with open(_file_name, 'r') as f:
        loaded_json = json.load(f)

    loaded_doc = Document.from_dict(loaded_json)
    assert loaded_doc._text_original == original_doc._text_original
    assert loaded_doc._text_cleaned == original_doc._text_cleaned
    for loaded_t, original_t in zip(loaded_doc._tokens, original_doc._tokens):
        _assert_tokens_equal(loaded_t, original_t)


def test__corpus__to_dict(corpus_simple_example):
    def _create_file_name(index: int) -> str:
        return get_test_file_path(f'spacy/corpus__to_doc_dicts__simple_example__{index}.json')

    # this is how I would save to file for large corpuses
    for index, doc_dict in enumerate(corpus_simple_example.to_doc_dicts()):
        with open(_create_file_name(index=index), 'w') as f:
            # json.dump() method writes the JSON string directly to the file object, so we don't
            # have to create the JSON string first.
            json.dump(doc_dict, f)

    def read_json_files():
        for index in range(len(corpus_simple_example)):
            with open(_create_file_name(index=index), 'r') as f:
                yield json.load(f)

    # recreate the original Corpus with a new object
    stop_words_to_add = {'dear', 'regard'}
    stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}
    new_corpus = Corpus(
        stop_words_to_add=stop_words_to_add,
        stop_words_to_remove=stop_words_to_remove,
        pre_process=clean,
        spacy_model='en_core_web_md',
        sklearn_tokenenizer_min_df=1,
    )
    new_corpus.from_doc_dicts(read_json_files())

    # for each doc test all attributes of document are the same and retest all tokens match
    for original_doc, new_doc in zip(corpus_simple_example, new_corpus):
        assert original_doc._text_original == new_doc._text_original
        assert original_doc._text_cleaned == new_doc._text_cleaned
        assert (original_doc.token_embeddings() == new_doc.token_embeddings()).all()
        assert (original_doc.embeddings() == new_doc.embeddings()).all()
        for loaded_t, original_t in zip(new_doc._tokens, original_doc._tokens):
            _assert_tokens_equal(loaded_t, original_t)

    ####
    # Lets make sure the vectorizers/etc. produce the same results
    ####
    orig_vector = corpus_simple_example.\
        text_to_count_vector("This is an awesome document with characters").\
        toarray()
    assert orig_vector.sum() > 1
    new_vector = new_corpus.\
        text_to_count_vector("This is an awesome document with characters").\
        toarray()
    assert (orig_vector == new_vector).all()

    orig_vector = corpus_simple_example.\
        text_to_tf_idf_vector("This is an awesome document with characters").\
        toarray()
    assert orig_vector.sum() > 0
    new_vector = new_corpus.\
        text_to_tf_idf_vector("This is an awesome document with characters").\
        toarray()
    assert (orig_vector == new_vector).all()

    assert (corpus_simple_example.embeddings_matrix() == new_corpus.embeddings_matrix()).all()
    assert (corpus_simple_example.count_vectorizer_vocab() == new_corpus.count_vectorizer_vocab()).all()  # noqa
    assert (corpus_simple_example.tf_idf_vectorizer_vocab() == new_corpus.tf_idf_vectorizer_vocab()).all()  # noqa
    assert (corpus_simple_example.similarity_matrix(how='embeddings-tf_idf') == new_corpus.similarity_matrix(how='embeddings-tf_idf')).all()  # noqa
    assert (corpus_simple_example.calculate_similarities(text='This is an awesome document', how='embeddings-tf_idf') == new_corpus.calculate_similarities(text='This is an awesome document', how='embeddings-tf_idf')).all()  # noqa


def test__corpus__count_lemmas():
    """
    This test will also test the `group_by` parameter, but using only one group to ensure that the
    same counts are returned.
    """
    documents = [
        "This is a document it really is a document; this is sort of important",
        "This is the #2 document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)

    lemma_count = corpus.count_lemmas(important_only=True, min_count=2, count_once_per_doc=False)
    assert lemma_count.set_index('token').to_dict()['count'] == {'document': 3, '_number_': 2}
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=True,
        min_count=2,
        count_once_per_doc=False
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa

    lemma_count = corpus.count_lemmas(important_only=True, min_count=2, count_once_per_doc=True)
    assert lemma_count.set_index('token').to_dict()['count'] == {'document': 2, '_number_': 2}
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=True,
        min_count=2,
        count_once_per_doc=True
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa

    lemma_count = corpus.count_lemmas(important_only=True, min_count=1, count_once_per_doc=False)
    assert lemma_count.set_index('token').to_dict()['count'] == {
        'document': 3,
        '_number_': 2,
        'sort': 1,
        'important': 1,
        '#': 1,
        'doc': 1,
    }
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=True,
        min_count=1,
        count_once_per_doc=False
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa

    lemma_count = corpus.count_lemmas(important_only=True, min_count=1, count_once_per_doc=True)
    assert lemma_count.set_index('token').to_dict()['count'] == {
        'document': 2,
        '_number_': 2,
        'sort': 1,
        'important': 1,
        '#': 1,
        'doc': 1,
    }
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=True,
        min_count=1,
        count_once_per_doc=True
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa

    lemma_count = corpus.count_lemmas(important_only=False, min_count=1, count_once_per_doc=False)
    assert lemma_count.set_index('token').to_dict()['count'] == {
        'be': 5, 'this': 4, 'document': 3, 'a': 2, 'the': 2, '_number_': 2, '.': 2, 'it': 1,
        'really': 1, 'sort': 1, 'of': 1, 'important': 1, 'doc': 1
    }
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=False,
        min_count=1,
        count_once_per_doc=False
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa

    lemma_count = corpus.count_lemmas(important_only=False, min_count=1, count_once_per_doc=True)
    assert lemma_count.set_index('token').to_dict()['count'] == {
        'this': 3, 'be': 3, 'document': 2, '_number_': 2, '.': 2, 'the': 2, 'it': 1, 'a': 1,
        'sort': 1, 'important': 1, 'of': 1, 'really': 1, 'doc': 1
    }
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_count = corpus.count_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=False,
        min_count=1,
        count_once_per_doc=True
    )
    assert (group_count['group'] == 'a').all()
    assert (group_count['count'] == group_count['token_count']).all()
    group_count_values = group_count['group_count'].unique()
    assert len(group_count_values) == 1
    assert (group_count.drop(columns=['group', 'token_count', 'group_count']) == lemma_count).all().all()  # noqa


def test__corpus__count_lemmas__group_by():
    documents = [
        "This is a doc. It's not really important. Shazam."
        "This is a document it really is a document; this is sort of important",
        "This is the #2 document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)
    lemma_count = corpus.count_lemmas(
        important_only=True,
        min_count=2,
        group_by=['a', 'b', 'b', 'a'],
        count_once_per_doc=False
    )
    assert (lemma_count['token_count'] >= 2).all()
    assert lemma_count.groupby('group')['group_count'].unique().apply(lambda x: len(x) == 1).all()
    expected_dict = {
        'group': {0: 'a', 1: 'a', 2: 'a', 3: 'b', 4: 'b', 5: 'b'},
        'token': {0: 'document', 1: 'important', 2: 'doc', 3: '_number_', 4: 'doc', 5: 'document'},
        'count': {0: 2, 1: 2, 2: 1, 3: 2, 4: 1, 5: 1},
        'token_count': {0: 3, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3},
        'group_count': {0: 7, 1: 7, 2: 7, 3: 5, 4: 5, 5: 5}
    }
    assert lemma_count.to_dict() == expected_dict

    lemma_count = corpus.count_lemmas(
        important_only=True,
        min_count=1,
        group_by=['a', 'b', 'b', 'a'],
        count_once_per_doc=False
    )
    assert (lemma_count['token_count'] >= 1).all()
    assert lemma_count.groupby('group')['group_count'].unique().apply(lambda x: len(x) == 1).all()
    expected_dict = {
        'group': {0: 'a', 1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'b', 6: 'b', 7: 'b', 8: 'b'},
        'token': {
            0: 'document', 1: 'important', 2: 'doc', 3: 'shazam', 4: 'sort',
            5: '_number_', 6: '#', 7: 'doc', 8: 'document'
        },
        'count': {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 2, 6: 1, 7: 1, 8: 1},
        'token_count': {0: 3, 1: 2, 2: 2, 3: 1, 4: 1, 5: 2, 6: 1, 7: 2, 8: 3},
        'group_count': {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 5, 6: 5, 7: 5, 8: 5}
    }
    assert lemma_count.to_dict() == expected_dict


def test__corpus__count_n_grams():
    documents = [
        "This is an important document it really is an important document; it's not an unimportant document.",  # noqa
        "This is the #2 important document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        stop_words_to_add={'_number_'},
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)
    bi_gram_count = corpus.count_n_grams(n=2, min_count=2, count_once_per_doc=False)
    assert bi_gram_count.set_index('token').to_dict()['count'] == {'important-document': 3}

    bi_gram_count = corpus.count_n_grams(n=2, separator='|', min_count=1, count_once_per_doc=False)
    assert bi_gram_count.set_index('token').to_dict()['count'] == {'important|document': 3, 'unimportant|document': 1}  # noqa

    bi_gram_count = corpus.count_n_grams(n=2, separator='|', min_count=1, count_once_per_doc=True)
    assert bi_gram_count.set_index('token').to_dict()['count'] == {'important|document': 2, 'unimportant|document': 1}  # noqa

    tri_gram_count = corpus.count_n_grams(n=3, min_count=1, count_once_per_doc=False)
    assert len(tri_gram_count) == 0


def test__corpus__tf_idf_lemmas():
    documents = [
        "This is a document it really is a document; this is sort of important",
        "This is the #2 document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)

    lemma_tf_idf = corpus.tf_idf_lemmas(important_only=True, min_count=1)
    expected_values = {'document': 3, '#': 1, 'doc': 1, 'important': 1, 'sort': 1, '_number_': 2}  # noqa
    assert lemma_tf_idf.set_index('token').to_dict()['count'] == expected_values
    expected_values = {'document': 1.516, '#': 1.199, 'doc': 1.199, 'important': 1.199, 'sort': 1.199, '_number_': 1.011}  # noqa
    assert lemma_tf_idf.set_index('token').round(3).to_dict()['tf_idf'] == expected_values
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_tf_idf = corpus.tf_idf_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=True,
        min_count=1,
    )
    assert (group_tf_idf.drop(columns=['group', 'token_count', 'group_count']) == lemma_tf_idf).all().all()  # noqa
    assert (group_tf_idf['token_count'] == lemma_tf_idf['count']).all()
    assert (group_tf_idf['group_count'] == lemma_tf_idf['count'].sum()).all()

    group_tf_idf = corpus.tf_idf_lemmas(
        group_by=['a', 'b', 'a'],
        important_only=True,
        min_count=1,
    )
    expected_values = {
        'group': {0: 'b', 1: 'a', 2: 'a', 3: 'a', 4: 'a', 5: 'a', 6: 'b', 7: 'b'},
        'token': {0: '#', 1: 'doc', 2: 'important', 3: 'sort', 4: 'document', 5: '_number_', 6: '_number_', 7: 'document'},  # noqa
        'count': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 1},
        'tf_idf': {0: 1.199, 1: 1.199, 2: 1.199, 3: 1.199, 4: 1.011, 5: 0.505, 6: 0.505, 7: 0.505},
        'token_count': {0: 1, 1: 1, 2: 1, 3: 1, 4: 3, 5: 2, 6: 2, 7: 3},
        'group_count': {0: 3, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 3, 7: 3}
    }
    assert group_tf_idf.round(3).to_dict() == expected_values
    assert (group_tf_idf.groupby('token')['token_count'].nunique() == 1).all()
    assert (group_tf_idf.groupby('group')['group_count'].nunique() == 1).all()

    lemma_tf_idf = corpus.tf_idf_lemmas(important_only=False, min_count=2)
    expected_values = {'a': 2, 'document': 3, '.': 2, '_number_': 2, 'the': 2, 'be': 5, 'this': 4}
    assert lemma_tf_idf.set_index('token').to_dict()['count'] == expected_values
    expected_values = {'a': 2.397, 'document': 1.516, '.': 1.011, '_number_': 1.011, 'the': 1.011, 'be': 0.5, 'this': 0.4}  # noqa
    assert lemma_tf_idf.set_index('token').round(3).to_dict()['tf_idf'] == expected_values
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_tf_idf = corpus.tf_idf_lemmas(
        group_by=['a', 'a', 'a'],
        important_only=False,
        min_count=2,
    )
    assert (group_tf_idf.drop(columns=['group', 'token_count', 'group_count']) == lemma_tf_idf).all().all()  # noqa
    assert (group_tf_idf.groupby('token')['token_count'].nunique() == 1).all()
    assert (group_tf_idf.groupby('group')['group_count'].nunique() == 1).all()

    group_tf_idf = corpus.tf_idf_lemmas(
        group_by=['a', 'b', 'a'],
        important_only=False,
        min_count=2,
    )
    expected_values = {
        'group': {0: 'a', 1: 'a', 2: 'a', 3: 'b', 4: 'a', 5: 'b', 6: 'b', 7: 'a', 8: 'b', 9: 'a', 10: 'a', 11: 'b', 12: 'b'},  # noqa
        'token': {0: 'a', 1: 'document', 2: '.', 3: '.', 4: '_number_', 5: '_number_', 6: 'document', 7: 'the', 8: 'the', 9: 'be', 10: 'this', 11: 'be', 12: 'this'},  # noqa
        'count': {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 4, 10: 3, 11: 1, 12: 1},  # noqa
        'tf_idf': {0: 2.397, 1: 1.011, 2: 0.505, 3: 0.505, 4: 0.505, 5: 0.505, 6: 0.505, 7: 0.505, 8: 0.505, 9: 0.4, 10: 0.3, 11: 0.1, 12: 0.1},  # noqa
        'token_count': {0: 2, 1: 3, 2: 2, 3: 2, 4: 2, 5: 2, 6: 3, 7: 2, 8: 2, 9: 5, 10: 4, 11: 5, 12: 4},  # noqa
        'group_count': {0: 20, 1: 20, 2: 20, 3: 6, 4: 20, 5: 6, 6: 6, 7: 20, 8: 6, 9: 20, 10: 20, 11: 6, 12: 6}  # noqa
    }
    assert group_tf_idf.round(3).to_dict() == expected_values
    assert (group_tf_idf.groupby('token')['token_count'].nunique() == 1).all()
    assert (group_tf_idf.groupby('group')['group_count'].nunique() == 1).all()


def test__corpus__tf_idf_n_grams():
    documents = [
        "This is a document it really is a document; this is sort of important",
        "This is the #2 document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)

    n_gram_tf_idf = corpus.tf_idf_n_grams(n=2, min_count=1)
    expected_values = {'_number_-doc': 1, '_number_-document': 1}
    assert n_gram_tf_idf.set_index('token').to_dict()['count'] == expected_values
    expected_values = {'_number_-doc': 1.199, '_number_-document': 1.199}
    assert n_gram_tf_idf.set_index('token').round(3).to_dict()['tf_idf'] == expected_values
    # this group count should be equivalent to the non-group dataframe since we pass in one group
    group_tf_idf = corpus.tf_idf_n_grams(
        group_by=['a', 'a', 'a'],
        n=2,
        min_count=1,
    )
    assert (group_tf_idf.drop(columns=['group', 'token_count', 'group_count']) == n_gram_tf_idf).all().all()  # noqa
    assert (group_tf_idf['token_count'] == n_gram_tf_idf['count']).all()
    assert (group_tf_idf['group_count'] == n_gram_tf_idf['count'].sum()).all()

    group_tf_idf = corpus.tf_idf_n_grams(
        group_by=['a', 'b', 'a'],
        n=2,
        min_count=1,
    )
    expected_values = {
        'group': {0: 'a', 1: 'b'},
        'token': {0: '_number_-doc', 1: '_number_-document'},
        'count': {0: 1, 1: 1},
        'tf_idf': {0: 1.199, 1: 1.199},
        'token_count': {0: 1, 1: 1},
        'group_count': {0: 1, 1: 1}
    }
    assert group_tf_idf.round(3).to_dict() == expected_values
    assert (group_tf_idf.groupby('token')['token_count'].nunique() == 1).all()
    assert (group_tf_idf.groupby('group')['group_count'].nunique() == 1).all()


def test__corpus__count_tokens():
    documents = [
        "This is a document it really is a document; this 1 document.",
        "This is a funny document.",
        "This is the # 3 doc."
    ]
    corpus = Corpus(
        pre_process=clean,
        spacy_model='en_core_web_sm',
    )
    corpus.fit(documents=documents)
    assert len(corpus.count_tokens(token_type='nouns', min_count=1)) > 0
    assert len(corpus.count_tokens(token_type='noun_phrases', min_count=1)) > 0
    assert len(corpus.count_tokens(token_type='adjectives_verbs', min_count=1)) > 0
    assert len(corpus.count_tokens(token_type='entities', min_count=1)) > 0
    assert len(corpus.tf_idf_tokens(token_type='nouns', min_count=1)) > 0
    assert len(corpus.tf_idf_tokens(token_type='noun_phrases', min_count=1)) > 0
    assert len(corpus.tf_idf_tokens(token_type='adjectives_verbs', min_count=1)) > 0
    assert len(corpus.tf_idf_tokens(token_type='entities', min_count=1)) > 0


def test__corpus__attributes(corpus_simple_example):
    # these are stupid tests but I just want to verify they run
    assert [x.sentiment() for x in corpus_simple_example] == list(corpus_simple_example.sentiments())  # noqa
    assert [x.impurity(original=True) for x in corpus_simple_example] == list(corpus_simple_example.impurities(original=True))  # noqa
    assert [x.impurity(original=False) for x in corpus_simple_example] == list(corpus_simple_example.impurities(original=False))  # noqa
    assert [len(x.text(original=True)) for x in corpus_simple_example] == list(corpus_simple_example.text_lengths(original=True))  # noqa
    assert [len(x.text(original=False)) for x in corpus_simple_example] == list(corpus_simple_example.text_lengths(original=False))  # noqa
    assert [x.num_tokens(important_only=True) for x in corpus_simple_example] == list(corpus_simple_example.num_tokens(important_only=True))  # noqa
    assert [x.num_tokens(important_only=False) for x in corpus_simple_example] == list(corpus_simple_example.num_tokens(important_only=False))  # noqa


def test__corpus__to_dataframe(corpus_simple_example, documents_fake):
    expected_columns = [
        'text_original',
        'text_clean',
        'lemmas_important',
        'bi_grams',
    ]
    df = corpus_simple_example.to_dataframe()
    assert df.columns.tolist() == expected_columns
    assert df['text_original'].tolist() == documents_fake
    assert df['text_clean'].notna().any()
    assert df['lemmas_important'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_important'])
    assert df['bi_grams'].notna().any()
    assert all(isinstance(x, list) for x in df['bi_grams'])

    expected_columns = [
        'text_clean',
        'text_original',
        'lemmas_important',
        'lemmas_all',
        'nouns',
        'bi_grams',
        'adjective_verbs',
        'noun_phrases',
        'impurity_original',
        'impurity_clean',
        'sentiment',
        'text_original_length',
        'text_clean_length',
        'num_tokens_important_only',
        'num_tokens_all',
    ]
    df = corpus_simple_example.to_dataframe(columns=expected_columns)
    assert df.columns.tolist() == expected_columns
    assert df['text_original'].tolist() == documents_fake
    assert df['text_clean'].notna().any()
    assert df['lemmas_important'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_important'])
    assert df['lemmas_all'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_all'])
    assert df['nouns'].notna().any()
    assert all(isinstance(x, list) for x in df['nouns'])
    assert df['bi_grams'].notna().any()
    assert all(isinstance(x, list) for x in df['bi_grams'])
    assert df['adjective_verbs'].notna().any()
    assert all(isinstance(x, list) for x in df['adjective_verbs'])
    assert df['noun_phrases'].notna().any()
    assert all(isinstance(x, list) for x in df['noun_phrases'])
    assert df['impurity_original'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_original'].fillna(0))
    assert df['impurity_clean'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_clean'].fillna(0))
    assert df['sentiment'].notna().any()
    assert all(isinstance(x, float) for x in df['sentiment'].fillna(0))
    assert df['text_original_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_original_length'].fillna(0))
    assert df['text_clean_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_clean_length'].fillna(0))
    assert df['num_tokens_important_only'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_important_only'].fillna(0))
    assert df['num_tokens_all'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_all'].fillna(0))

    expected_columns = [
        'text_original',
        'text_clean',
        'lemmas_all',
        'lemmas_important',
        'bi_grams',
        'nouns',
        'noun_phrases',
        'adjective_verbs',
        'sentiment',
        'impurity_original',
        'impurity_clean',
        'text_clean_length',
        'text_original_length',
        'num_tokens_all',
        'num_tokens_important_only',
    ]
    df = corpus_simple_example.to_dataframe(columns='all')
    assert df.columns.tolist() == expected_columns
    assert df['text_original'].tolist() == documents_fake
    assert df['text_clean'].notna().any()
    assert df['lemmas_important'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_important'])
    assert df['lemmas_all'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_all'])
    assert df['nouns'].notna().any()
    assert all(isinstance(x, list) for x in df['nouns'])
    assert df['bi_grams'].notna().any()
    assert all(isinstance(x, list) for x in df['bi_grams'])
    assert df['adjective_verbs'].notna().any()
    assert all(isinstance(x, list) for x in df['adjective_verbs'])
    assert df['noun_phrases'].notna().any()
    assert all(isinstance(x, list) for x in df['noun_phrases'])
    assert df['impurity_original'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_original'].fillna(0))
    assert df['impurity_clean'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_clean'].fillna(0))
    assert df['sentiment'].notna().any()
    assert all(isinstance(x, float) for x in df['sentiment'].fillna(0))
    assert df['text_original_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_original_length'].fillna(0))
    assert df['text_clean_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_clean_length'].fillna(0))
    assert df['num_tokens_important_only'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_important_only'].fillna(0))
    assert df['num_tokens_all'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_all'].fillna(0))


def test__corpus__to_dataframe__first_n(corpus_simple_example, documents_fake):
    df_full = corpus_simple_example.to_dataframe(columns='all')
    expected_columns = [
        'text_original',
        'text_clean',
        'lemmas_all',
        'lemmas_important',
        'bi_grams',
        'nouns',
        'noun_phrases',
        'adjective_verbs',
        'sentiment',
        'impurity_original',
        'impurity_clean',
        'text_clean_length',
        'text_original_length',
        'num_tokens_all',
        'num_tokens_important_only',
    ]
    df = corpus_simple_example.to_dataframe(columns='all', first_n=2)
    assert len(df) == 2
    assert (df_full.iloc[0:2].isna() == df.isna()).all().all()
    assert (df_full.iloc[0:2].notna() == df.notna()).all().all()
    assert (df_full.iloc[0:2].fillna(-10) == df.fillna(-10)).all().all()
    assert df.columns.tolist() == expected_columns
    assert df['text_original'].tolist() == documents_fake[0:2]
    assert df['text_clean'].notna().any()
    assert df['lemmas_important'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_important'])
    assert df['lemmas_all'].notna().any()
    assert all(isinstance(x, list) for x in df['lemmas_all'])
    assert df['nouns'].notna().any()
    assert all(isinstance(x, list) for x in df['nouns'])
    assert df['bi_grams'].notna().any()
    assert all(isinstance(x, list) for x in df['bi_grams'])
    assert df['adjective_verbs'].notna().any()
    assert all(isinstance(x, list) for x in df['adjective_verbs'])
    assert df['noun_phrases'].notna().any()
    assert all(isinstance(x, list) for x in df['noun_phrases'])
    assert df['impurity_original'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_original'].fillna(0))
    assert df['impurity_clean'].notna().any()
    assert all(isinstance(x, float) for x in df['impurity_clean'].fillna(0))
    assert df['sentiment'].notna().any()
    assert all(isinstance(x, float) for x in df['sentiment'].fillna(0))
    assert df['text_original_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_original_length'].fillna(0))
    assert df['text_clean_length'].notna().any()
    assert all(isinstance(x, int) for x in df['text_clean_length'].fillna(0))
    assert df['num_tokens_important_only'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_important_only'].fillna(0))
    assert df['num_tokens_all'].notna().any()
    assert all(isinstance(x, int) for x in df['num_tokens_all'].fillna(0))


def test__corpus__diff(corpus_simple_example):
    corpus = corpus_simple_example

    with open(get_test_file_path('spacy/document_diff__sample_1.html'), 'w') as file:
        file.write(corpus[1].diff())

    with open(get_test_file_path('spacy/document_diff__sample_1__use_lemmas.html'), 'w') as file:
        file.write(corpus[1].diff(use_lemmas=True))

    with open(get_test_file_path('spacy/corpus_diff__sample.html'), 'w') as file:
        file.write(corpus.diff())

    with open(get_test_file_path('spacy/corpus_diff__sample__first_2.html'), 'w') as file:
        file.write(corpus.diff(first_n=2))

    with open(get_test_file_path('spacy/corpus_diff__sample__use_lemmas.html'), 'w') as file:
        file.write(corpus.diff(use_lemmas=True))


def test__corpus__embeddings(corpus_simple_example):
    corpus = corpus_simple_example
    non_empty_corpi = [corpus[1], corpus[3], corpus[5]]

    assert len(corpus[0].embeddings()) == 0
    assert len(corpus[0].token_embeddings()) == 0

    embedding_shape = corpus[1][0].embeddings.shape
    assert embedding_shape[0] > 0
    assert corpus[1].embeddings().shape == embedding_shape
    assert all([d.token_embeddings().shape  == (d.num_tokens(important_only=True), embedding_shape[0]) for d in non_empty_corpi])  # noqa
    assert all((d.embeddings() > 0).any() for d in non_empty_corpi)

    expected_embeddings_length = corpus[1][0].embeddings.shape[0]
    embeddings_average = corpus.embeddings_matrix(aggregation='average')
    assert embeddings_average.shape == (len(corpus), expected_embeddings_length)
    # same test to maek sure lru_cache plays nice with generators
    assert (corpus.embeddings_matrix(aggregation='average') == embeddings_average).all()
    assert (embeddings_average[0] == 0).all()
    assert (embeddings_average[2] == 0).all()
    assert (embeddings_average[4] == 0).all()
    assert (embeddings_average[1] != 0).any()
    assert (embeddings_average[3] != 0).any()
    assert (embeddings_average[5] != 0).any()
    with open(get_test_file_path('spacy/corpus__embeddings_matrix__average.txt'), 'w') as file:
        _T = embeddings_average.T
        for index in range(len(_T)):
            file.write(str(_T[index]) + "\n")

    # tf-idf embeddings
    embeddings_tf_idf = corpus.embeddings_matrix(aggregation='tf_idf')
    assert embeddings_tf_idf.shape == embeddings_average.shape
    # same test to maek sure lru_cache plays nice with generators
    assert (corpus.embeddings_matrix(aggregation='tf_idf') == embeddings_tf_idf).all()
    assert (embeddings_tf_idf[0] == 0).all()
    assert (embeddings_tf_idf[2] == 0).all()
    assert (embeddings_tf_idf[4] == 0).all()
    assert (embeddings_tf_idf[1] != 0).any()
    assert (embeddings_tf_idf[3] != 0).any()
    assert (embeddings_tf_idf[5] != 0).any()
    with open(get_test_file_path('spacy/corpus__embeddings_matrix__tf_idf.txt'), 'w') as file:
        _T = embeddings_tf_idf.T
        for index in range(len(_T)):
            file.write(str(_T[index]) + "\n")


def test__corpus__prepare_doc_for_vectorizer(corpus_simple_example, documents_fake):
    corpus = corpus_simple_example
    assert corpus._max_tokens is None
    assert corpus._max_bi_grams is None

    doc = corpus._text_to_doc(text=documents_fake[1])
    # `_text_to_doc` should produce the same document (which implies the same embeddings)
    # as the original
    assert (doc.embeddings() == corpus[1].embeddings()).all()
    assert list(doc.lemmas(important_only=False)) == list(corpus[1].lemmas(important_only=False))
    assert corpus._prepare_doc_for_vectorizer(doc) == corpus._prepare_doc_for_vectorizer(corpus[1])
    assert corpus._prepare_doc_for_vectorizer(doc) == '_number_ document important information # character _number_-document important-information'  # noqa

    # now change the max-tokens and max-bigrams and make sure we get the expected result from
    # both our custom doc and the original in the corpus
    corpus._max_tokens = 2
    corpus._max_bi_grams = 1
    assert corpus._prepare_doc_for_vectorizer(doc) == '_number_ document _number_-document'
    assert corpus._prepare_doc_for_vectorizer(doc) == corpus._prepare_doc_for_vectorizer(corpus[1])


def test__corpus__vectorizers(corpus_simple_example, documents_fake):
    corpus = corpus_simple_example
    ####
    # Test Count/Vectors
    ####
    assert corpus.count_matrix().shape[0] == len(corpus)
    count_matrix = corpus.count_matrix().toarray()
    assert (count_matrix > 0).any()
    assert len(corpus.count_vectorizer_vocab()) == corpus.count_matrix().shape[1]
    _count_df = pd.DataFrame(count_matrix)
    _count_df.columns = corpus.count_vectorizer_vocab()
    dataframe_to_text_file(
        _count_df.transpose(),
        get_test_file_path('spacy/corpus__count_matrix__sample.txt')
    )
    empty_text_vector = corpus.text_to_count_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.count_vectorizer_vocab()), )
    assert (empty_text_vector == 0).all()
    # text text_to_count_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_count_vector(text=documents_fake[i])
        vector = vector.toarray()[0]
        assert vector.shape == count_matrix[i].shape
        assert all(vector == count_matrix[i])

    ####
    # Test TF-IDF/Vectors
    ####
    assert corpus.tf_idf_matrix().shape[0] == len(corpus)
    assert corpus.count_matrix().shape == corpus.tf_idf_matrix().shape
    assert len(corpus.tf_idf_vectorizer_vocab()) == corpus.tf_idf_matrix().shape[1]
    assert (corpus.count_vectorizer_vocab() == corpus.tf_idf_vectorizer_vocab()).all()

    tf_idf_matrix = corpus.tf_idf_matrix().toarray()
    assert (tf_idf_matrix > 0).any()
    _tf_idf_df = pd.DataFrame(tf_idf_matrix)
    _tf_idf_df.columns = corpus.tf_idf_vectorizer_vocab()
    dataframe_to_text_file(
        _tf_idf_df.transpose(),
        get_test_file_path('spacy/corpus__tf_idf_matrix__sample.txt')
    )
    empty_text_vector = corpus.text_to_tf_idf_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.tf_idf_vectorizer_vocab()), )
    assert (empty_text_vector == 0).all()
    # text text_to_tf_idf_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_tf_idf_vector(text=documents_fake[i])
        vector = vector.toarray()[0]
        assert vector.shape == tf_idf_matrix[i].shape
        assert (vector.round(5) == tf_idf_matrix[i].round(5)).all()


def test__corpus__similarity_matrix(corpus_simple_example):
    corpus = corpus_simple_example

    ####
    # Test Similarity Matrix
    ####
    def _check_sim_matrix(sim_matrix):
        assert (sim_matrix[0] == 0).all()
        assert (sim_matrix[:, 0] == 0).all()
        assert (sim_matrix[2] == 0).all()
        assert (sim_matrix[:, 2] == 0).all()
        assert (sim_matrix[4] == 0).all()
        assert (sim_matrix[:, 4] == 0).all()
        # diagnals of non-empty docs
        assert (sim_matrix[[1, 3, 5], [1, 3, 5]].round(5) == 1).all()

    sim_matrix_emb_average = corpus.similarity_matrix(how='embeddings-average')
    assert sim_matrix_emb_average.shape == (len(corpus), len(corpus))
    _check_sim_matrix(sim_matrix_emb_average)
    with open(get_test_file_path('spacy/corpus__similarity_matrix__embeddings_average.html'), 'w') as file:  # noqa
        file.write(str(sim_matrix_emb_average))

    sim_matrix_emb_tf_idf = corpus.similarity_matrix(how='embeddings-tf_idf')
    assert sim_matrix_emb_tf_idf.shape == (len(corpus), len(corpus))
    _check_sim_matrix(sim_matrix_emb_tf_idf)
    assert (sim_matrix_emb_average.round(5) != sim_matrix_emb_tf_idf.round(5)).any()
    with open(get_test_file_path('spacy/corpus__similarity_matrix__embeddings_tf_idf.html'), 'w') as file:  # noqa
        file.write(str(sim_matrix_emb_tf_idf))

    sim_matrix_count = corpus.similarity_matrix(how='count')
    assert sim_matrix_count.shape == (len(corpus), len(corpus))
    _check_sim_matrix(sim_matrix_count)
    assert (sim_matrix_count.round(5) != sim_matrix_emb_average.round(5)).any()
    assert (sim_matrix_count.round(5) != sim_matrix_emb_tf_idf.round(5)).any()
    with open(get_test_file_path('spacy/corpus__similarity_matrix__count.html'), 'w') as file:  # noqa
        file.write(str(sim_matrix_count))

    sim_matrix_tf_idf = corpus.similarity_matrix(how='tf_idf')
    assert sim_matrix_tf_idf.shape == (len(corpus), len(corpus))
    _check_sim_matrix(sim_matrix_tf_idf)
    assert (sim_matrix_tf_idf.round(5) != sim_matrix_count.round(5)).any()
    assert (sim_matrix_tf_idf.round(5) != sim_matrix_emb_average.round(5)).any()
    assert (sim_matrix_tf_idf.round(5) != sim_matrix_emb_tf_idf.round(5)).any()
    with open(get_test_file_path('spacy/corpus__similarity_matrix__tf_idf.html'), 'w') as file:  # noqa
        file.write(str(sim_matrix_tf_idf))


def test__corpus__calculate_similarities(corpus_simple_example, documents_fake):
    corpus = corpus_simple_example

    ####
    # calculate_similarity
    ####
    def _test_calculate_similarity(how: str):
        results = corpus.calculate_similarities(text='', how=how)
        assert results.shape == (len(corpus), )
        assert (results == 0).all()

        results = corpus.calculate_similarities(text=documents_fake[1], how=how)
        assert results.shape == (len(corpus), )
        assert results[1].round(5) == 1
        assert 0 < results[3] < 1
        assert 0 < results[5] < 1
        assert results[0] == 0
        assert results[2] == 0
        assert results[4] == 0

        results = corpus.calculate_similarities(text=documents_fake[3], how=how)
        assert results.shape == (len(corpus), )
        assert results[3].round(5) == 1
        assert 0 < results[1] < 1
        assert 0 < results[5] < 1
        assert results[0] == 0
        assert results[2] == 0
        assert results[4] == 0

        results = corpus.calculate_similarities(text=documents_fake[5], how=how)
        assert results.shape == (len(corpus), )
        assert results[5].round(5) == 1
        assert 0 < results[3] < 1
        assert 0 < results[1] < 1
        assert results[0] == 0
        assert results[2] == 0
        assert results[4] == 0

    _test_calculate_similarity(how='count')
    _test_calculate_similarity(how='tf_idf')
    _test_calculate_similarity(how='embeddings-average')
    _test_calculate_similarity(how='embeddings-tf_idf')


def test__corpus__get_similar_doc_indexes(corpus_simple_example, documents_fake):  # noqa
    corpus = corpus_simple_example

    def _test_get_similar_doc_indexes(how: str):
        indexes, similarities = corpus.get_similar_doc_indexes(0, how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(2, how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(4, how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0

        indexes, similarities = corpus.get_similar_doc_indexes(1, how=how)
        assert (indexes != 1).all()
        assert len(indexes) == 2
        assert len(similarities) == 2
        assert list(indexes) == [3, 5]
        (similarities > 0).all()
        (similarities < 1).all()
        indexes, similarities = corpus.get_similar_doc_indexes(3, how=how)
        assert (indexes != 3).all()
        assert len(indexes) == 2
        assert len(similarities) == 2
        assert list(indexes) == [1, 5]
        (similarities > 0).all()
        (similarities < 1).all()
        indexes, similarities = corpus.get_similar_doc_indexes(5, how=how)
        assert (indexes != 5).all()
        assert len(indexes) == 2
        assert len(similarities) == 2
        if how == 'embeddings-average':
            assert list(indexes) == [3, 1]
        else:
            assert list(indexes) == [1, 3]
        (similarities > 0).all()
        (similarities < 1).all()

    _test_get_similar_doc_indexes(how='count')
    _test_get_similar_doc_indexes(how='tf_idf')
    _test_get_similar_doc_indexes(how='embeddings-average')
    _test_get_similar_doc_indexes(how='embeddings-tf_idf')

    def _test_get_similar_doc_indexes(how: str):
        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[0], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[2], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[4], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0

        # now the index should contain the doc with the equivalent index because it is the same
        # doc, but we passed it in as text so the class doesn't know it is the same doc;
        # it should have a similarity of 1
        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[1], how=how)
        # same text was used so the best index should be the same as in documents_fake
        assert indexes[0] == 1
        assert len(indexes) == 3
        assert len(similarities) == 3
        assert list(indexes) == [1, 3, 5]
        assert (similarities > 0).all()
        assert (similarities.round(5) <= 1).all()
        # same text was used so the highest similarity should be 1 i.e. same text
        assert similarities[0].round(5) == 1

        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[3], how=how)
        # same text was used so the best index should be the same as in documents_fake
        assert indexes[0] == 3
        assert len(indexes) == 3
        assert len(similarities) == 3
        assert list(indexes) == [3, 1, 5]
        assert (similarities > 0).all()
        assert (similarities.round(5) <= 1).all()
        # same text was used so the highest similarity should be 1 i.e. same text
        assert similarities[0].round(5) == 1

        indexes, similarities = corpus.get_similar_doc_indexes(documents_fake[5], how=how)
        # same text was used so the best index should be the same as in documents_fake
        assert indexes[0] == 5
        assert len(indexes) == 3
        assert len(similarities) == 3
        if how == 'embeddings-average':
            assert list(indexes) == [5, 3, 1]
        else:
            assert list(indexes) == [5, 1, 3]
        assert (similarities > 0).all()
        assert (similarities.round(5) <= 1).all()
        # same text was used so the highest similarity should be 1 i.e. same text
        assert similarities[0].round(5) == 1

    _test_get_similar_doc_indexes(how='count')
    _test_get_similar_doc_indexes(how='tf_idf')
    _test_get_similar_doc_indexes(how='embeddings-average')
    _test_get_similar_doc_indexes(how='embeddings-tf_idf')


def test__corpus__tokens__reddit(corpus_reddit, reddit):
    reddit = reddit.copy()
    corpus = corpus_reddit
    # check that the original text in each document in the corpus matches the original post (i.e.
    # in the same order)
    assert all(d.text(original=True) == r for d, r in zip(corpus, reddit['post']))

    with open(get_test_file_path('spacy/corpus__text__original__reddit.txt'), 'w') as handle:
        handle.writelines(t + "\n" for t in corpus.text())

    with open(get_test_file_path('spacy/corpus__text__clean__reddit.txt'), 'w') as handle:
        handle.writelines(t + "\n" for t in corpus.text(original=False))

    with open(get_test_file_path('spacy/corpus__lemmas__important_only__reddit.txt'), 'w') as handle:  # noqa
        handle.writelines('|'.join(x) + "\n" for x in corpus.lemmas())

    with open(get_test_file_path('spacy/corpus__lemmas__important_only__false__reddit.txt'), 'w') as handle:  # noqa
        handle.writelines('|'.join(x) + "\n" for x in corpus.lemmas(important_only=False))

    with open(get_test_file_path('spacy/corpus__n_grams__2__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.n_grams(2))

    with open(get_test_file_path('spacy/corpus__n_grams__3__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.n_grams(3, separator='--'))

    with open(get_test_file_path('spacy/corpus__nouns__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.nouns())

    with open(get_test_file_path('spacy/corpus__noun_phrases__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.noun_phrases())

    with open(get_test_file_path('spacy/corpus__adjectives_verbs__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(x) + "\n" for x in corpus.adjectives_verbs())

    with open(get_test_file_path('spacy/corpus__entities__reddit.txt'), 'w') as handle:
        handle.writelines('|'.join(f"{e[0]} ({e[1]})" for e in x) + "\n" for x in corpus.entities())  # noqa


def test__corpus__diff__reddit(corpus_reddit):
    corpus = corpus_reddit

    with open(get_test_file_path('spacy/document_diff__reddit_1.html'), 'w') as file:
        file.write(corpus[1].diff())

    with open(get_test_file_path('spacy/document_diff__reddit_1__use_lemmas.html'), 'w') as file:
        file.write(corpus[1].diff(use_lemmas=True))

    with open(get_test_file_path('spacy/corpus_diff__reddit.html'), 'w') as file:
        file.write(corpus.diff())

    with open(get_test_file_path('spacy/corpus_diff__reddit__first_2.html'), 'w') as file:
        file.write(corpus.diff(first_n=2))

    with open(get_test_file_path('spacy/corpus_diff__reddit__use_lemmas.html'), 'w') as file:
        file.write(corpus.diff(use_lemmas=True))


def test__corpus__embeddings__reddit(corpus_reddit):
    corpus = corpus_reddit

    embedding_shape = corpus[0][0].embeddings.shape
    assert embedding_shape[0] > 0
    assert corpus[1].embeddings().shape == embedding_shape
    assert all([d.token_embeddings().shape  == (d.num_tokens(important_only=True), embedding_shape[0]) for d in corpus])  # noqa
    assert all((d.embeddings() > 0).any() for d in corpus)

    expected_embeddings_length = corpus[1][0].embeddings.shape[0]
    embeddings_average = corpus.embeddings_matrix(aggregation='average')
    assert embeddings_average.shape == (len(corpus), expected_embeddings_length)
    # same test to maek sure lru_cache plays nice with generators
    assert (corpus.embeddings_matrix(aggregation='average') == embeddings_average).all()

    # tf-idf embeddings
    embeddings_tf_idf = corpus.embeddings_matrix(aggregation='tf_idf')
    assert embeddings_tf_idf.shape == embeddings_average.shape
    # same test to maek sure lru_cache plays nice with generators
    assert (corpus.embeddings_matrix(aggregation='tf_idf') == embeddings_tf_idf).all()

    # documents.term_freq()
    # documents.plot_wordcloud()
    # documents.search()
    # documents.similarity_matrix()
    # documents.find_similar()


def test__corpus__vectorizers__reddit(corpus_reddit, reddit):
    reddit = reddit.copy()
    corpus = corpus_reddit
    ####
    # Test Count/Vectors
    ####
    assert corpus.count_matrix().shape[0] == len(corpus)
    count_matrix = corpus.count_matrix().toarray()
    assert (count_matrix > 0).any()
    assert len(corpus.count_vectorizer_vocab()) == corpus.count_matrix().shape[1]
    empty_text_vector = corpus.text_to_count_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.count_vectorizer_vocab()), )
    assert (empty_text_vector == 0).all()
    # text text_to_count_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    reddit_post_list = reddit['post'].tolist().copy()
    for i in range(len(corpus)):
        vector = corpus.text_to_count_vector(text=reddit_post_list[i])
        vector = vector.toarray()[0]
        assert vector.shape == count_matrix[i].shape
        assert all(vector == count_matrix[i])

    ####
    # Test TF-IDF/Vectors
    ####
    assert corpus.tf_idf_matrix().shape[0] == len(corpus)
    assert corpus.count_matrix().shape == corpus.tf_idf_matrix().shape
    assert len(corpus.tf_idf_vectorizer_vocab()) == corpus.tf_idf_matrix().shape[1]
    assert (corpus.count_vectorizer_vocab() == corpus.tf_idf_vectorizer_vocab()).all()

    tf_idf_matrix = corpus.tf_idf_matrix().toarray()
    assert (tf_idf_matrix > 0).any()
    _tf_idf_df = pd.DataFrame(tf_idf_matrix)
    _tf_idf_df.columns = corpus.tf_idf_vectorizer_vocab()
    empty_text_vector = corpus.text_to_tf_idf_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.tf_idf_vectorizer_vocab()), )
    assert (empty_text_vector == 0).all()
    # text text_to_tf_idf_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_tf_idf_vector(text=reddit_post_list[i])
        vector = vector.toarray()[0]
        assert vector.shape == tf_idf_matrix[i].shape
        assert (vector.round(5) == tf_idf_matrix[i].round(5)).all()


def test__Corpus_num_batches(corpus_reddit, corpus_reddit_parallel):
    assert len(corpus_reddit) == len(corpus_reddit_parallel)
    # check that the original/clean text in each document for both corpuses matches the original
    # post (i.e ensure the same order and same cleaning)
    assert all(
        np.text(original=True) == p.text(original=True)
        for np, p in zip(corpus_reddit, corpus_reddit_parallel)
    )
    assert all(
        np.text(original=False) == p.text(original=False)
        for np, p in zip(corpus_reddit, corpus_reddit_parallel)
    )
    assert all(
        list(np.lemmas(important_only=False)) == list(p.lemmas(important_only=False))
        for np, p in zip(corpus_reddit, corpus_reddit_parallel)
    )
    assert all(
        (np.embeddings() == p.embeddings()).all()
        for np, p in zip(corpus_reddit, corpus_reddit_parallel)
    )
