from math import ceil
import logging
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

from source.library.text_analysis import impurity
from source.library.text_cleaning_simple import prepare, get_n_grams, get_stop_words, tokenize
from source.library.text_preparation import clean, predict_language
from source.library.spacy import create_spacy_pipeline, custom_tokenizer, extract_from_doc
from source.library.utilities import create_batch_start_stop_indexes


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
        un_debates.to_pickle('artifacts/data/raw/un-general-debates-blueprint.pkl')

    with Timer("Loading Reddit Dataset - Saving to /artifacts/data/raw/reddit.pkl"):
        logging.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/reddit-selfposts")  # noqa
        reddit = pd.read_csv('artifacts/data/external/reddit.tsv.zip', sep="\t")
        reddit.rename(columns={'selftext': 'post'}, inplace=True)
        reddit.to_pickle('artifacts/data/raw/reddit.pkl')


def processing_text_data(df: pd.DataFrame):
    df['speaker'].fillna('<unknown>', inplace=True)
    df['position'].fillna('<unknown>', inplace=True)
    assert not df.isna().any().any()
    df['text_length'] = df['text'].str.len()
    df['tokens'] = df['text'].apply(prepare)
    df['num_tokens'] = df['tokens'].map(len)
    df['bi_grams'] = df['text'].\
        apply(prepare, pipeline=[str.lower, tokenize]).\
        apply(get_n_grams, n=2, stop_words=get_stop_words())
    df['num_bi_grams'] = df['bi_grams'].map(len)

    return (df)


def reddit_extract_from_doc(df: pd.DataFrame) -> pd.DataFrame:
    nlp = create_spacy_pipeline(
        stopwords_to_add={'dear', 'regards'},
        stopwords_to_remove={'down'},
        tokenizer=custom_tokenizer
    )
    docs = nlp.pipe(df['post_clean'])
    for j, doc in enumerate(docs):
        for col, values in extract_from_doc(doc).items():
            df[col].iloc[j] = values
    return df


@main.command()
@log_function_call
@log_timer
def transform():
    """
    Transforms the data.
    """
    ####
    # UN Debate Data
    ####
    with Timer("Loading UN Generate Debate Dataset"):
        un_debates = pd.read_pickle('artifacts/data/raw/un-general-debates-blueprint.pkl')

    batch_size = 500
    num_batches = ceil(len(un_debates) / batch_size)
    batch_indexes = create_batch_start_stop_indexes(
        length=len(un_debates),
        num_batches=num_batches
    )

    datasets = [un_debates.iloc[x[0]:x[1]].copy() for x in batch_indexes]
    assert sum([len(x) for x in datasets]) == len(un_debates)

    with Timer("UN Debate - Processing Text Data"):
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(processing_text_data, datasets))
            debates_transformed = pd.concat(results)
            assert len(debates_transformed) == len(un_debates)
            un_debates = debates_transformed
            del debates_transformed, datasets, batch_size, num_batches, batch_indexes
            assert not un_debates.isna().any().any()

    with Timer("Saving UN-Debaates"):
        un_debates.to_pickle('artifacts/data/processed/un-general-debates-blueprint.pkl')

    with Timer("Creating UN dataset that is Per Year/Country/Paragraph"):
        paragraphs_series = un_debates["text"].map(lambda text: regex.split(r'\.\s*\n', text))
        # flatten the paragraphs keeping the years, countries
        un_debates_paragraphs = pd.DataFrame(
            [
                {"year": year, "country": country, "text": paragraph}
                for year, country, paragraphs in zip(un_debates["year"],
                                                     un_debates["country_name"],
                                                     paragraphs_series)
                for paragraph in paragraphs if paragraph
            ]
        )
        un_debates_paragraphs['text'] = un_debates_paragraphs['text'].str.strip()
        un_debates_paragraphs = un_debates_paragraphs[un_debates_paragraphs['text'] != '']

    # This takes about 30 minutes to run.
    # with Timer("Cleaning UN Paragraph dataset "):
    #     nlp = create_spacy_pipeline(stopwords_to_add={'dear', 'regards'},
    #                                 stopwords_to_remove={'down'},
    #                                 tokenizer=custom_tokenizer)
    #
    #     def text_lemmas(text: str) -> str:
    #         text = clean(
    #             text,
    #             remove_angle_bracket_content=True,
    #             remove_bracket_content=True,
    #         )
    #         lemmas = extract_from_doc(
    #             doc=nlp(text),
    #             all_lemmas=True,
    #             partial_lemmas=False,
    #             bi_grams=False,
    #             adjectives_verbs=False,
    #             nouns=False,
    #             noun_phrases=False,
    #             named_entities=False
    #         )
    #         return ' '.join(lemmas['all_lemmas'])
    #
    #     un_debates_paragraphs['clean_text'] = None
    #     batch_size = 25000
    #     from math import ceil
    #     num_batches = ceil(len(un_debates_paragraphs) / batch_size)
    #     for i in range(0, len(un_debates_paragraphs), batch_size):
    #         logging.info(f"Processing Batch {round((i / batch_size) + 1)} of {num_batches}")
    #         results = [text_lemmas(x) for x in un_debates_paragraphs['text'][i:i + batch_size]]
    #         for j, result in enumerate(results):
    #             un_debates_paragraphs['clean_text'].iloc[i + j] = result

    message = "Saving UN Paragraphs dataset to " \
        "/artifacts/data/processed/un-general-debates-paragraphs.pkl"
    with Timer(message=message):
        un_debates_paragraphs.to_pickle(
            'artifacts/data/processed/un-general-debates-paragraphs.pkl'
        )

    ####
    # Processing Reddit Data
    ####
    with Timer("Loading Reddit Dataset - Sampling 5K rows."):
        reddit = pd.read_pickle('artifacts/data/raw/reddit.pkl')
        reddit = reddit.sample(5000, random_state=42)

    with Timer("Loading Reddit Dataset - Calculating Impurity"):
        reddit['impurity'] = reddit['post'].apply(impurity)

    with Timer("Cleaning Data"):
        reddit['post_clean'] = reddit['post'].apply(clean, remove_bracket_content=False)
    assert not reddit.isna().any().any()
    with Timer("Tokenizing & Extracting"):
        nlp = create_spacy_pipeline(
            stopwords_to_add={'dear', 'regards'},
            stopwords_to_remove={'down'},
            tokenizer=custom_tokenizer
        )
        nlp_columns = list(extract_from_doc(nlp.make_doc('')).keys())
        for col in nlp_columns:
            reddit[col] = None

        batch_size = 500
        num_batches = ceil(len(reddit) / batch_size)
        batch_indexes = create_batch_start_stop_indexes(
            length=len(reddit),
            num_batches=num_batches
        )
        datasets = [reddit.iloc[x[0]:x[1]].copy() for x in batch_indexes]
        assert sum([len(x) for x in datasets]) == len(reddit)

        with ProcessPoolExecutor() as pool:
            # temp = reddit_extract_from_doc(df=datasets[0])
            results = list(pool.map(reddit_extract_from_doc, datasets))
            reddit_transformed = pd.concat(results)
            assert len(reddit_transformed) == len(reddit)
            added_columns = {
                'all_lemmas', 'partial_lemmas', 'bi_grams', 'adjs_verbs', 'nouns', 'noun_phrases',
                'entities'
            }
            assert added_columns.issubset(set(reddit_transformed.columns))
            reddit = reddit_transformed
            del reddit_transformed, datasets, batch_size, num_batches, batch_indexes
            assert not reddit.isna().any().any()

        # for i in range(0, len(reddit), batch_size):
        #     logging.info(f"Processing Batch {round((i / batch_size) + 1)} of {num_batches}")
        #     docs = nlp.pipe(reddit['post_clean'][i:i + batch_size])
        #     for j, doc in enumerate(docs):
        #         for col, values in extract_from_doc(doc).items():
        #             reddit[col].iloc[i + j] = values

        reddit['post_length'] = reddit['post'].str.len()
        reddit['num_tokens'] = reddit['partial_lemmas'].map(len)

    assert not reddit.isna().any().any()

    language_model = fasttext.load_model("/fasttext/lid.176.ftz")
    with Timer("Reddit - Predicting Language"):
        reddit['language'] = reddit['post'].apply(predict_language, model=language_model)
        logging.info(
            f"Language was not able to be determined on {reddit['language'].isna().sum()} records."
        )

    with Timer("Saving processed Reddit dataset to /artifacts/data/processed/reddit.pkl"):
        reddit.to_pickle('artifacts/data/processed/reddit.pkl')


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
            stop_words=stop_words,
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
            stop_words=stop_words,
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
            stop_words=stop_words,
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
        k_means_model = KMeans(n_clusters=10, random_state=42)
        k_means_model.fit(tfidf_vectors)
        file = f'artifacts/models/topics/k_means-topics-{num_topics}-ngrams-{ngrams_low}-{ngrams_high}__model.pkl'  # noqa
        logging.info(f"Saving K-Means model: {file}")
        to_pickle(k_means_model, file)


if __name__ == '__main__':
    main()
