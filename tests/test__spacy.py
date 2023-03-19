import pandas as pd

from source.library.text_preparation import clean, predict_language
from source.library.spacy import Corpus
from tests.helpers import dataframe_to_text_file, get_test_file_path


def test__DocumentProcessor__simple():
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

    docs_str = [
        '',
        '<b>This is 1 document that has very important information</b> and some $ & # characters.',
        'I',
        'This is another (the 2nd i.e. 2) unimportant document with almost no #awesome information!!',  # noqa
        '  ',
        "However, this is 'another' document with hyphen-word. & It has two sentences and is dumb."
    ]
    corpus.fit(documents=docs_str)
    assert len(corpus) == len(docs_str)
    # make sure these didn't reset during fitting
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)

    # first let's check the documents with edge cases
    assert corpus[0].text(original=True) == ''
    assert corpus[0].text(original=False) == ''
    assert corpus[2].text(original=True) == 'I'
    assert corpus[2].text(original=False) == 'I'
    assert corpus[4].text(original=True) == '  '
    assert corpus[4].text(original=False) == ''

    assert corpus[0].num_important_tokens() == 0
    assert corpus[2].num_important_tokens() == 0
    assert corpus[4].num_important_tokens() == 0

    assert list(corpus[0].lemmas(important_only=True)) == []
    assert list(corpus[0].lemmas(important_only=False)) == []
    assert list(corpus[2].lemmas(important_only=True)) == []
    assert list(corpus[2].lemmas(important_only=False)) == ['i']
    assert list(corpus[4].lemmas(important_only=True)) == []
    assert list(corpus[4].lemmas(important_only=False)) == []

    ####
    # Test Document functionality
    ####
    with open(get_test_file_path(f'spacy/document_processor__text.txt'), 'w') as handle:  # noqa
        handle.writelines([d.text(original=False) + "\n" for d in corpus])

    for index in range(len(docs_str)):
        assert [t.text for t in corpus[index]] == [t.text for t in corpus[index]._tokens]
        assert len(corpus[index]) == len(corpus[index]._tokens)
        dataframe_to_text_file(
            pd.DataFrame(corpus[index].to_dict()),
            get_test_file_path(f'spacy/document_to_dict__sample_{index}.txt')
        )

    non_empty_corpi = [corpus[1], corpus[3], corpus[5]]
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

    ####
    # Test Corpus functionality
    ####
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

    ####
    # Test Embeddings
    ####
    assert len(corpus[0].embeddings()) == 0
    assert len(corpus[0].token_embeddings()) == 0

    embedding_shape = corpus[1][0].embeddings.shape
    assert embedding_shape[0] > 0
    assert corpus[1].embeddings().shape == embedding_shape
    assert all([d.token_embeddings().shape  == (d.num_important_tokens(), embedding_shape[0]) for d in non_empty_corpi])  # noqa
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

    ####
    # Test Count/Vectors
    ####
    assert corpus.count_matrix().shape[0] == len(corpus)
    count_matrix = corpus.count_matrix().toarray()
    assert (count_matrix > 0).any()
    assert len(corpus.count_vocabulary()) == corpus.count_matrix().shape[1]
    _count_df = pd.DataFrame(count_matrix)
    _count_df.columns = corpus.count_vocabulary()
    dataframe_to_text_file(
        _count_df.transpose(),
        get_test_file_path('spacy/corpus__count_matrix__sample.txt')
    )
    empty_text_vector = corpus.text_to_count_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.count_vocabulary()), )
    assert (empty_text_vector == 0).all()
    # text text_to_count_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_count_vector(text=docs_str[i])
        vector = vector.toarray()[0]
        assert vector.shape == count_matrix[i].shape
        assert all(vector == count_matrix[i])

    ####
    # Test TF-IDF/Vectors
    ####
    assert corpus.tf_idf_matrix().shape[0] == len(corpus)
    assert corpus.count_matrix().shape == corpus.tf_idf_matrix().shape
    assert len(corpus.tf_idf_vocabulary()) == corpus.tf_idf_matrix().shape[1]
    assert (corpus.count_vocabulary() == corpus.tf_idf_vocabulary()).all()

    tf_idf_matrix = corpus.tf_idf_matrix().toarray()
    assert (tf_idf_matrix > 0).any()
    _tf_idf_df = pd.DataFrame(tf_idf_matrix)
    _tf_idf_df.columns = corpus.tf_idf_vocabulary()
    dataframe_to_text_file(
        _tf_idf_df.transpose(),
        get_test_file_path('spacy/corpus__tf_idf_matrix__sample.txt')
    )
    empty_text_vector = corpus.text_to_tf_idf_vector(text='').toarray()[0]
    assert empty_text_vector.shape == (len(corpus.tf_idf_vocabulary()), )
    assert (empty_text_vector == 0).all()
    # text text_to_tf_idf_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_tf_idf_vector(text=docs_str[i])
        vector = vector.toarray()[0]
        assert vector.shape == tf_idf_matrix[i].shape
        assert (vector.round(5) == tf_idf_matrix[i].round(5)).all()

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

    ####
    # calculate_similarity
    ####
    def _test_calculate_similarity(how: str):
        results = corpus.calculate_similarities(text='', how=how)
        assert results.shape == (len(corpus), )
        assert (results == 0).all()

        results = corpus.calculate_similarities(text=docs_str[1], how=how)
        assert results.shape == (len(corpus), )
        assert results[1].round(5) == 1
        assert 0 < results[3] < 1
        assert 0 < results[5] < 1
        assert results[0] == 0
        assert results[2] == 0
        assert results[4] == 0

        results = corpus.calculate_similarities(text=docs_str[3], how=how)
        assert results.shape == (len(corpus), )
        assert results[3].round(5) == 1
        assert 0 < results[1] < 1
        assert 0 < results[5] < 1
        assert results[0] == 0
        assert results[2] == 0
        assert results[4] == 0

        results = corpus.calculate_similarities(text=docs_str[5], how=how)
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
        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[0], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[2], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0
        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[4], how=how)
        assert len(indexes) == 0
        assert len(similarities) == 0

        # now the index should contain the doc with the equivalent index because it is the same
        # doc, but we passed it in as text so the class doesn't know it is the same doc;
        # it should have a similarity of 1
        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[1], how=how)
        # same text was used so the best index should be the same as in docs_str
        assert indexes[0] == 1
        assert len(indexes) == 3
        assert len(similarities) == 3
        assert list(indexes) == [1, 3, 5]
        assert (similarities > 0).all()
        assert (similarities.round(5) <= 1).all()
        # same text was used so the highest similarity should be 1 i.e. same text
        assert similarities[0].round(5) == 1

        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[3], how=how)
        # same text was used so the best index should be the same as in docs_str
        assert indexes[0] == 3
        assert len(indexes) == 3
        assert len(similarities) == 3
        assert list(indexes) == [3, 1, 5]
        assert (similarities > 0).all()
        assert (similarities.round(5) <= 1).all()
        # same text was used so the highest similarity should be 1 i.e. same text
        assert similarities[0].round(5) == 1

        indexes, similarities = corpus.get_similar_doc_indexes(docs_str[5], how=how)
        # same text was used so the best index should be the same as in docs_str
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

    # Test Corpus functionality

# def test__DocumentProcessor__reddit(reddit):

#     # Test Token specific characteristics
    
#     docs_str = reddit.apply(lambda x: x['title'] + ' ... ' + x['post'], axis=1)
#     documents = processor.process(documents=docs_str)
#     assert len(documents) == len(reddit)


#     # Test Document specific characteristics
#     assert str(documents[0]) == documents[0].text


#     documents[0][0].__dict__.keys()


    # # Test Token specific characteristics


    # str(documents[0])
    # documents[0][0].text
    # reddit['title']


    # documents.term_freq_inverse_doc_freq()
    # documents.term_freq()
    # documents.plot_wordcloud()
    # documents.search()
    # documents.similarity_matrix()
    # documents.find_similar()


    # post_a = documents[0]
    # post_a.sentiment


