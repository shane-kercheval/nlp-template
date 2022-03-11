import click
import pandas as pd

from helpers.utilities import get_logger, Timer


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





if __name__ == '__main__':
    main()
