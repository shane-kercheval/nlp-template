import pandas as pd

from source.library.text_preparation import clean, predict_language
from source.library.spacy import Corpus
from tests.helpers import dataframe_to_text_file, get_test_file_path


def test__DocumentProcessor__simple():
    corpus = Corpus()
    assert 'dear' not in corpus.stop_words
    assert 'regard' not in corpus.stop_words
    assert 'down' in corpus.stop_words

    corpus = Corpus(
        stop_words_to_add={'dear', 'regard'},
        stop_words_to_remove={'down'},
        pre_process=clean,
        spacy_model='en_core_web_md'
    )
    assert 'dear' in corpus.stop_words
    assert 'regard' in corpus.stop_words
    assert 'down' not in corpus.stop_words

    docs_str = [
        '<b>This is 1 document that has very important information</b> and some $ & # characters.',
        'This is another (the 2nd i.e. 2) document with almost no #awesome information!!',
        "However, this is 'another' document with hyphen-word. It has two sentences and is dumb."
    ]
    corpus.fit(documents=docs_str)
    assert len(corpus) == len(docs_str)

    # Test Document functionality
    with open(get_test_file_path(f'spacy/document_processor__text.txt'), 'w') as handle:  # noqa
        handle.writelines([d.text + "\n" for d in corpus])

    for index in range(len(docs_str)):
        assert [x.text for x in corpus[index]] == [x.text for x in corpus[index]._tokens]
        assert len(corpus[index]) == len(corpus[index]._tokens)

        dataframe_to_text_file(
            pd.DataFrame(corpus[index].to_dict()),
            get_test_file_path(f'spacy/document_to_dict__sample_{index}.txt')
        )

    assert corpus[0].embeddings().shape == corpus[0][0].embeddings.shape
    assert all(len(d.lemmas()) > 0 for d in corpus)
    assert all(len(d.n_grams()) > 0 for d in corpus)
    assert all(len(d.nouns()) > 0 for d in corpus)
    assert all(len(d.adjectives_verbs()) > 0 for d in corpus)
    assert all(len(d.entities()) > 0 for d in corpus)
    assert all(len(d.n_grams()) > 0 for d in corpus)
    assert corpus[0].sentiment() == 0  # not working yet

    corpus.embeddings_matrix().shape

    assert corpus.count_matrix().shape[0] == len(corpus)
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


