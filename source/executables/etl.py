import click
import pandas as pd

from helpers.utilities import get_logger, Timer
from helpers.text_cleaning_simple import prepare, get_n_grams, get_stop_words, tokenize


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


    with Timer("Loading Reddit Dataset"):
        reddit = pd.read_pickle('artifacts/data/raw/reddit.pkl')

    #with Timer("Reddit - Processing Text Data"):
        
    assert not reddit.isna().any().any()
    with Timer("Saving processed Reddit dataset to /artifacts/data/processed/reddit.pkl"):
        reddit.to_pickle('artifacts/data/processed/reddit.pkl')


if __name__ == '__main__':
    main()
