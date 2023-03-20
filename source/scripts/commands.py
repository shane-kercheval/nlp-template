from math import ceil
import logging
import logging.config
import click
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
import fasttext

from helpsk.utility import to_pickle
from helpsk.logging import log_function_call, log_timer, Timer

import regex

from source.library.spacy import Corpus

from source.library.text_analysis import impurity
from source.library.text_cleaning_simple import prepare, get_n_grams, get_stop_words, tokenize
from source.library.text_preparation import clean, predict_language
from source.library.utilities import create_batch_start_stop_indexes
from source.library.datasets import DATA

stop_words = STOP_WORDS.copy()
stop_words |= {'united', 'nations', 'nation'}
stop_words |= {'ll', 've'}

logging.config.fileConfig(
    "source/config/logging_to_file.conf",
    defaults={'logfilename': 'output/log.log'},
    disable_existing_loggers=False
)


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
@log_function_call
@log_timer
def extract():
    """
    Extracts the data.
    """
    logging.info("Extracting Data")
    with Timer("Loading UN Generate Debate Dataset - Saving to /artifacts/data/raw/un-general-debates-blueprint.pkl"):  # noqa
        logging.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/un-general-debates")  # noqa
        un_debates = pd.read_csv('artifacts/data/external/un-general-debates-blueprint.csv.zip')
        DATA.un_debates.save(un_debates)

    with Timer("Loading Reddit Dataset - Saving to /artifacts/data/raw/reddit.pkl"):
        logging.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/reddit-selfposts")  # noqa
        reddit = pd.read_csv('artifacts/data/external/reddit.tsv.zip', sep="\t")
        reddit.rename(columns={'selftext': 'post'}, inplace=True)
        DATA.reddit.save(reddit)


@main.command()
@log_function_call
@log_timer
def transform():
    """
    Transforms the data.
    """
    ####
    # Reddit
    ####
    with Timer("Loading Reddit Dataset"):
        reddit = DATA.reddit.load()

    with Timer(f"Creating Reddit Corpus ({len(reddit):,} documents)"):
        stop_words_to_add = {'dear', 'regard', '_number_', '_tag_'}
        stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}
        corpus = Corpus(
            stop_words_to_add=stop_words_to_add,
            stop_words_to_remove=stop_words_to_remove,
            pre_process=clean,
            spacy_model='en_core_web_sm',
            sklearn_tokenenizer_min_df=20,
            sklearn_tokenenizer_max_tokens=200,
            sklearn_tokenenizer_max_bi_grams=30,
        )
        assert all(x in corpus.stop_words for x in stop_words_to_add)
        assert all(x not in corpus.stop_words for x in stop_words_to_remove)
        corpus.fit(documents=reddit['post'].tolist())
    DATA.reddit_corpus.save(corpus)

    ####
    # UN Debate Data
    ####
    with Timer("Loading UN Generate Debate Dataset"):
        un_debates = DATA.un_debates.load()

    with Timer("Creating UN dataset that is Per Year/Country/Paragraph"):
        paragraphs_series = un_debates["text"].map(lambda text: regex.split(r'\.\s*\n', text))
        # flatten the paragraphs keeping the years, countries
        un_debate_paragraphs = pd.DataFrame(
            [
                {"year": year, "country": country, "text": paragraph}
                for year, country, paragraphs in zip(un_debates["year"],
                                                     un_debates["country_name"],
                                                     paragraphs_series)
                for paragraph in paragraphs if paragraph
            ]
        )
        un_debate_paragraphs['text'] = un_debate_paragraphs['text'].str.strip()
        un_debate_paragraphs = un_debate_paragraphs[un_debate_paragraphs['text'] != '']
    DATA.un_debate_paragraphs.save(un_debate_paragraphs)

    # un_debate_paragraphs = un_debate_paragraphs.sample(50000)
    with Timer(f"Creating UN dataset Corpus ({len(un_debate_paragraphs):,} documents)"):
        stop_words_to_add = {'dear', 'regard', '_number_', '_tag_'}
        stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}
        corpus = Corpus(
            stop_words_to_add=stop_words_to_add,
            stop_words_to_remove=stop_words_to_remove,
            pre_process=clean,
            spacy_model='en_core_web_sm',
            sklearn_tokenenizer_min_df=20,
            sklearn_tokenenizer_max_tokens=200,
            sklearn_tokenenizer_max_bi_grams=30,
        )
        assert all(x in corpus.stop_words for x in stop_words_to_add)
        assert all(x not in corpus.stop_words for x in stop_words_to_remove)
        corpus.fit(documents=un_debate_paragraphs['text'].tolist())
    DATA.un_debate_corpus.save(corpus)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
@log_function_call
@log_timer
def nmf(num_topics, ngrams_low, ngrams_high, num_samples):
    """
    Run Non-negative Matrix Factorization.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logging.info("Running NMF.")
    logging.info(f"Number of Topics : {num_topics}")
    logging.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logging.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF-IDF"):
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        logging.info(f"Saving NMF vectorizer: {file}")
        to_pickle(tfidf_vectorizer, file)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logging.info(f"Saving NMF vectors: {file}")
        to_pickle(tfidf_vectors, file)

    with Timer("Running Non-Negative Matrix Factorization (NMF)"):  # noqa
        nmf_model = NMF(init='nndsvda', n_components=num_topics, random_state=42, max_iter=1000)
        _ = nmf_model.fit_transform(tfidf_vectors)
        file = f'artifacts/models/topics/nmf-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logging.info(f"Saving NMF model: {file}")
        to_pickle(nmf_model, file)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
@log_function_call
@log_timer
def lda(num_topics, ngrams_low, ngrams_high, num_samples):
    """
    Run Latent Dirichlet Allocation.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logging.info("Running LDA.")
    logging.info(f"Number of Topics : {num_topics}")
    logging.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logging.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF"):
        count_vectorizer = CountVectorizer(
            stop_words=list(stop_words),
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        count_vectors = count_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        logging.info(f"Saving LDA vectorizer: {file}")
        to_pickle(count_vectorizer, file)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logging.info(f"Saving LDA vectors: {file}")
        to_pickle(count_vectors, file)

    with Timer("Running Latent Dirichlet Allocation (LDA)"):  # noqa
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        _ = lda_model.fit_transform(count_vectors)
        file = f'artifacts/models/topics/lda-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logging.info(f"Saving LDA model: {file}")
        to_pickle(lda_model, file)


@main.command()
@click.option('-num_topics', default=10, show_default=True)
@click.option('-ngrams_low', default=1, show_default=True)
@click.option('-ngrams_high', default=3, show_default=True)
@click.option('-num_samples', default=5000, show_default=True)
@log_function_call
@log_timer
def k_means(num_topics, ngrams_low, ngrams_high, num_samples):
    """
    Run K-Means Clustering.

    Args:
        num_topics: int; number of topics to create
        ngrams_low: int; minimum number of grams (e.g. 1 is uni-grams)
        ngrams_high: int; maximum number of grams (e.g. 2 is bi-grams)
        num_samples: int; the number of samples to use when creating topics.
    """
    logging.info("Running K-Means.")
    logging.info(f"Number of Topics : {num_topics}")
    logging.info(f"N-Gram Range     : {ngrams_low}-{ngrams_high}")
    logging.info(f"Number of Samples: {num_samples}")

    with Timer("Loading Data"):
        path = 'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        paragraphs = pd.read_pickle(path)
        paragraphs = paragraphs.sample(num_samples, random_state=42)

    with Timer("Calculating TF-IDF"):
        tfidf_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            ngram_range=(ngrams_low, ngrams_high),
            min_df=5,
            max_df=0.7
        )
        tfidf_vectors = tfidf_vectorizer.fit_transform(paragraphs["text"])
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectorizer.pkl'  # noqa
        logging.info(f"Saving K-Means vectorizer: {file}")
        to_pickle(tfidf_vectorizer, file)
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__vectors.pkl'  # noqa
        logging.info(f"Saving K-Means vectors: {file}")
        to_pickle(tfidf_vectors, file)

    with Timer("Running K-Means Clustering"):  # noqa
        k_means_model = KMeans(n_clusters=10, n_init='auto', random_state=42)
        k_means_model.fit(tfidf_vectors)
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logging.info(f"Saving K-Means model: {file}")
        to_pickle(k_means_model, file)


if __name__ == '__main__':
    main()
