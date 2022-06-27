import os
import sys

import click
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
    """
    Run Non-negative Matrix Factorization.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logger = get_logger()
    logger.info("Running NMF.")
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
        logger.info(f"Saving NMF vectorizer: {file}")
        to_pickle(tfidf_vectorizer, file)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logger.info(f"Saving NMF vectors: {file}")
        to_pickle(tfidf_vectors, file)

    with Timer("Running Non-Negative Matrix Factorization (NMF)"):  # noqa
        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42, max_iter=1000)
        _ = nmf_model.fit_transform(tfidf_vectors)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logger.info(f"Saving NMF model: {file}")
        to_pickle(nmf_model, file)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
def lda(num_topics, ngrams_low, ngrams_high, num_samples):
    """
    Run Latent Dirichlet Allocation.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logger = get_logger()
    logger.info("Running LDA.")
    logger.info(f"Number of Topics : {num_topics}")
    logger.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logger.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF"):
        count_vectorizer = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        count_vectors = count_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        logger.info(f"Saving LDA vectorizer: {file}")
        to_pickle(count_vectorizer, file)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logger.info(f"Saving LDA vectors: {file}")
        to_pickle(count_vectors, file)

    with Timer("Running Latent Dirichlet Allocation (LDA)"):  # noqa
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        _ = lda_model.fit_transform(count_vectors)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logger.info(f"Saving LDA model: {file}")
        to_pickle(lda_model, file)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
def k_means(num_topics, ngrams_low, ngrams_high, num_samples):
    """
    Run K-Means Clustering.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logger = get_logger()
    logger.info("Running K-Means.")
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
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        logger.info(f"Saving K-Means vectorizer: {file}")
        to_pickle(tfidf_vectorizer, file)
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logger.info(f"Saving K-Means vectors: {file}")
        to_pickle(tfidf_vectors, file)

    with Timer("Running K-Means Clustering"):  # noqa
        k_means_model = KMeans(n_clusters=10, random_state=42)
        k_means_model.fit(tfidf_vectors)
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logger.info(f"Saving K-Means model: {file}")
        to_pickle(k_means_model, file)


if __name__ == '__main__':
    main()
