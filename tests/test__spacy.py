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
        '<b>This is 1 document that has very important information</b> and some $ & # characters.',
        'This is another (the 2nd i.e. 2) unimportant document with almost no #awesome information!!',  # noqa
        "However, this is 'another' document with hyphen-word. & It has two sentences and is dumb."
    ]
    corpus.fit(documents=docs_str)
    assert len(corpus) == len(docs_str)
    # make sure these didn't reset during fitting
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)

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

    embedding_shape = corpus[0][0].embeddings.shape
    assert corpus[0].embeddings().shape == embedding_shape
    assert all([d.token_embeddings().shape  == (d.num_important_tokens(), embedding_shape[0]) for d in corpus])  # noqa
    assert all((d.embeddings() > 0).any() for d in corpus)
    assert all(len(list(d.lemmas())) > 0 for d in corpus)
    assert all(len(list(d.n_grams())) > 0 for d in corpus)
    assert all(len(list(d.nouns())) > 0 for d in corpus)
    assert all(len(list(d.noun_phrases())) > 0 for d in corpus)
    assert all(len(list(d.adjectives_verbs())) > 0 for d in corpus)
    assert all(len(list(d.entities())) > 0 for d in corpus)
    assert corpus[0].sentiment() == 0  # not working yet

    # sanity check cache (e.g. which could fail with generators)
    assert corpus[0].impurity(original=False) == corpus[0].impurity(original=False)
    assert corpus[0].impurity(original=True) == corpus[0].impurity(original=True)
    assert all(d.impurity(original=True) > 0 for d in corpus)
    assert all(d.impurity(original=True) > d.impurity(original=False) for d in corpus)

    # sanity check cache (e.g. which could fail with generators)
    assert corpus[0].diff(use_lemmas=False) == corpus[0].diff(use_lemmas=False)
    assert corpus[0].diff(use_lemmas=True) == corpus[0].diff(use_lemmas=True)
    assert len(corpus[0].diff(use_lemmas=False)) > 0
    assert len(corpus[0].diff(use_lemmas=True)) > 0

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

    with open(get_test_file_path('spacy/document_diff__sample_1.html'), 'w') as file:
        file.write(corpus[0].diff())

    with open(get_test_file_path('spacy/document_diff__sample_1__use_lemmas.html'), 'w') as file:
        file.write(corpus[0].diff(use_lemmas=True))

    with open(get_test_file_path('spacy/corpus_diff__sample.html'), 'w') as file:
        file.write(corpus.diff())

    with open(get_test_file_path('spacy/corpus_diff__sample__first_2.html'), 'w') as file:
        file.write(corpus.diff(first_n=2))

    with open(get_test_file_path('spacy/corpus_diff__sample__use_lemmas.html'), 'w') as file:
        file.write(corpus.diff(use_lemmas=True))

    expected_embeddings_length = corpus[0][0].embeddings.shape[0]
    embeddings_matrix = corpus.embeddings_matrix()
    assert embeddings_matrix.shape == (len(corpus), expected_embeddings_length)
    # same test to maek sure lru_cache plays nice with generators
    assert (corpus.embeddings_matrix() == embeddings_matrix).all()
    assert (embeddings_matrix != 0).any()

    assert corpus.count_matrix().shape[0] == len(corpus)
    count_matrix = corpus.count_matrix().toarray()
    assert (count_matrix > 0).any()
    _count_df = pd.DataFrame(count_matrix)
    _count_df.columns = corpus.count_token_names()
    dataframe_to_text_file(
        _count_df.transpose(),
        get_test_file_path('spacy/corpus__count_matrix__sample.txt')
    )
    # text text_to_count_vector by confirming if we pass in the same text we originally passed in
    # then it will get processed and vectorized in the same way and therefore have the same vector
    # values
    for i in range(len(corpus)):
        vector = corpus.text_to_count_vector(text=docs_str[i])
        vector = vector.toarray()[0]
        assert vector.shape == count_matrix[i].shape
        assert all(vector == count_matrix[i])




    assert corpus.tf_idf_matrix().shape[0] == len(corpus)
    assert corpus.count_matrix().shape == corpus.tf_idf_matrix().shape
    corpus.count_matrix().toarray()

    corpus.count_token_names()
    corpus.count_matrix().sum(axis=0)[[0]]
    assert len(corpus.count_token_names()) == corpus.count_matrix().shape[1]

    corpus.tf_idf_token_names()
    corpus.tf_idf_matrix().sum(axis=0)
    assert len(corpus.tf_idf_token_names()) == corpus.tf_idf_matrix().shape[1]
    assert (corpus.count_token_names() == corpus.tf_idf_token_names()).all()

    corpus.tf_idf_matrix().toarray()
    corpus.tf_idf()

    with open(get_test_file_path('spacy/document_diff__sample_1.html'), 'w') as file:
        file.write(corpus[0].diff())

    with open(get_test_file_path('spacy/document_diff__sample_1__use_lemmas.html'), 'w') as file:
        file.write(corpus[0].diff(use_lemmas=True))

    with open(get_test_file_path('spacy/corpus_diff__sample.html'), 'w') as file:
        file.write(corpus.diff())

    with open(get_test_file_path('spacy/corpus_diff__sample__first_2.html'), 'w') as file:
        file.write(corpus.diff(first_n=2))

    with open(get_test_file_path('spacy/corpus_diff__sample__use_lemmas.html'), 'w') as file:
        file.write(corpus.diff(use_lemmas=True))

        

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


