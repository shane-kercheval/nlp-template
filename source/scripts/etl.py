import click
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from source.library.utilities import get_logger, Timer  # noqa
from source.library.text_analysis import impurity  # noqa
from source.library.text_cleaning_simple import prepare, get_n_grams, get_stop_words, tokenize  # noqa
from source.library.text_preparation import clean, predict_language  # noqa
from source.library.spacy import create_spacy_pipeline, custom_tokenizer, extract_from_doc, doc_to_dataframe  # noqa


@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    logger = get_logger()
    logger.info("Extracting Data")
    with Timer("Loading UN Generate Debate Dataset - Saving to /artifacts/data/raw/un-general-debates-blueprint.pkl"):
        logger.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/un-general-debates")  # noqa
        un_debates = pd.read_csv('artifacts/data/external/un-general-debates-blueprint.csv.zip')
        un_debates.to_pickle('artifacts/data/raw/un-general-debates-blueprint.pkl')

    with Timer("Loading Reddit Dataset - Saving to /artifacts/data/raw/reddit.pkl"):
        logger.info("This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/reddit-selfposts")  # noqa
        reddit = pd.read_csv('artifacts/data/external/reddit.tsv.zip', sep="\t")
        reddit.rename(columns={'selftext': 'post'}, inplace=True)
        reddit.to_pickle('artifacts/data/raw/reddit.pkl')


@main.command()
def transform():
    logger = get_logger()
    logger.info("Transforming Data")
    with Timer("Loading UN Generate Debate Dataset"):
        un_debates = pd.read_pickle('artifacts/data/raw/un-general-debates-blueprint.pkl')

    with Timer("UN Debate - Processing Text Data"):
        un_debates['speaker'].fillna('<unknown>', inplace=True)
        un_debates['position'].fillna('<unknown>', inplace=True)
        assert not un_debates.isna().any().any()
        un_debates['text_length'] = un_debates['text'].str.len()
        logger.info("Generating tokens a.k.a uni-grams.")
        un_debates['tokens'] = un_debates['text'].apply(prepare)
        un_debates['num_tokens'] = un_debates['tokens'].map(len)
        logger.info("Generating bi-grams.")
        un_debates['bi_grams'] = un_debates['text'].\
            apply(prepare, pipeline=[str.lower, tokenize]).\
            apply(get_n_grams, n=2, stop_words=get_stop_words())
        un_debates['num_bi_grams'] = un_debates['bi_grams'].map(len)

    assert not un_debates.isna().any().any()
    with Timer("Saving processed UN Debate dataset to /artifacts/data/processed/un-general-debates-blueprint.pkl"):
        un_debates.to_pickle('artifacts/data/processed/un-general-debates-blueprint.pkl')

    with Timer("Loading Reddit Dataset - Sampling 5K rows."):
        reddit = pd.read_pickle('artifacts/data/raw/reddit.pkl')
        reddit = reddit.sample(5000, random_state=42)

    with Timer("Loading Reddit Dataset - Calculating Impurity"):
        reddit['impurity'] = reddit['post'].apply(impurity)

    with Timer("Reddit - Cleaning Data"):
        reddit['post_clean'] = reddit['post'].apply(clean,
                                                    remove_bracket_content=False)
    assert not reddit.isna().any().any()
    with Timer("Reddit - Tokenizing & Extracting"):
        nlp = create_spacy_pipeline(stopwords_to_add={'dear', 'regards'},
                                    stopwords_to_remove={'down'},
                                    tokenizer=custom_tokenizer)

        nlp_columns = list(extract_from_doc(nlp.make_doc('')).keys())
        for col in nlp_columns:
            reddit[col] = None

        batch_size = 50
        for i in range(0, len(reddit), batch_size):
            docs = nlp.pipe(reddit['post_clean'][i:i + batch_size])
            for j, doc in enumerate(docs):
                for col, values in extract_from_doc(doc).items():
                    reddit[col].iloc[i + j] = values

        reddit['post_length'] = reddit['post'].str.len()
        reddit['num_tokens'] = reddit['partial_lemmas'].map(len)

    assert not reddit.isna().any().any()

    import fasttext
    language_model = fasttext.load_model("source/resources/lid.176.ftz")
    with Timer("Reddit - Predicting Language"):
        reddit['language'] = reddit['post'].apply(predict_language, model=language_model)
        logger.info(f"Language was not able to be determined on {reddit['language'].isna().sum()} records.")

    with Timer("Saving processed Reddit dataset to /artifacts/data/processed/reddit.pkl"):
        reddit.to_pickle('artifacts/data/processed/reddit.pkl')


if __name__ == '__main__':
    main()
