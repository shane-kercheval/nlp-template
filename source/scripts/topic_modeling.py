import os
import sys

import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS

from helpsk.utility import to_pickle

sys.path.append(os.getcwd())
from source.library.utilities import get_logger, Timer  # noqa
from source.library.text_analysis import impurity  # noqa
from source.library.text_cleaning_simple import prepare, get_n_grams, get_stop_words, tokenize  # noqa
from source.library.text_preparation import clean, predict_language  # noqa
from source.library.spacy import create_spacy_pipeline, custom_tokenizer, extract_from_doc  # noqa

stop_words = STOP_WORDS.copy()
stop_words |= {'united', 'nations', 'nation'}
stop_words |= {'ll', 've'}


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
def nmf(num_topics, ngrams_low, ngrams_high, num_samples):
    logger = get_logger()
    logger.info(f"Running NMF.")
    logger.info(f"Number of Topics : {num_topics}")
    logger.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logger.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF-IDF"):
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        to_pickle(tfidf_vectorizer, file)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        to_pickle(tfidf_vectors, file)

    with Timer("Running Non-Negative Matrix Factorization (NMF)"):  # noqa
        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42, max_iter=1000)
        _ = nmf_model.fit_transform(tfidf_vectors)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        to_pickle(nmf_model, file)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
def lda(num_topics, ngrams_low, ngrams_high, num_samples):
    logger = get_logger()
    logger.info(f"Running LDA.")
    logger.info(f"Number of Topics : {num_topics}")
    logger.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logger.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF-IDF"):
        count_vectorizer = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        count_vectors = count_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        to_pickle(count_vectorizer, file)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        to_pickle(count_vectors, file)

    with Timer("Running Latent Dirichlet Allocation (LDA)"):  # noqa
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        _ = lda_model.fit_transform(count_vectors)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        to_pickle(lda_model, file)


if __name__ == '__main__':
    main()