import click
import pandas as pd

from helpers.utilities import get_logger, Timer
from helpers.text_processing import prepare


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


@main.command()
def transform():
    logger = get_logger()
    logger.info("Transforming Data")
    with Timer("Loading UN Generate Debate Dataset"):
        un_debates = pd.read_pickle('artifacts/data/raw/un-general-debates-blueprint.pkl')

    with Timer("Processing Text Data"):
        un_debates['speaker'].fillna('<unknown>', inplace=True)
        un_debates['position'].fillna('<unknown>', inplace=True)
        assert not un_debates.isna().any().any()
        un_debates['tokens'] = un_debates['text'].apply(prepare)
        un_debates['num_tokens'] = un_debates['tokens'].map(len)
        un_debates['text_length'] = un_debates['text'].str.len()

    assert not un_debates.isna().any().any()
    with Timer("Saving processed UN Debate dataset to /artifacts/data/processed/un-general-debates-blueprint.pkl"):
        un_debates.to_pickle('artifacts/data/processed/un-general-debates-blueprint.pkl')

if __name__ == '__main__':
    main()
