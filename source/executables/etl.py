import pickle
import warnings

import click
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from helpers.utilities import get_logger
    

@click.group()
def main():
    """
    Logic For Extracting and Transforming Datasets
    """
    pass


@main.command()
def extract():
    logger = get_logger()
    logger.info(f"Downloading credit data from https://www.openml.org/d/31")
    credit_g = fetch_openml('credit-g', version=1)
    credit_data = credit_g['data']
    credit_data['target'] = credit_g['target']
    logger.info(f"Credit data downloaded with {credit_data.shape[0]} rows and {credit_data.shape[1]} columns.")
    
    # Create Missing Values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        credit_data['duration'].iloc[0:46] = np.nan
        credit_data['checking_status'].iloc[25:75] = np.nan
        credit_data['credit_amount'].iloc[10:54] = 0
    
    logger.info(f"Done processing credit data.")

    output_file = 'artifacts/data/raw/credit.pkl'
    logger.info(f"Saving credit data to `{output_file}`")
    credit_data.to_pickle(output_file)


@main.command()
def transform():
    logger = get_logger()
    logger.info(f"Transforming Data")


@main.command()
def create_training_test():
    logger = get_logger()
    logger.info(f"Splitting training & test datasets")

    with open('artifacts/data/raw/credit.pkl', 'rb') as handle:
        credit_data = pickle.load(handle)

    y_full = credit_data['target']
    x_full = credit_data.drop(columns='target')

    # i.e. value of 0 is 'good' i.e. 'not default' and value of 1 is bad and what
    # we want to detect i.e. 'default'
    y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=42)

    file_name = 'artifacts/data/processed/x_train.pkl'
    logger.info(f"Creating `{file_name}`")
    with open(file_name, 'wb') as handle:
        pickle.dump(x_train, handle)

    file_name = 'artifacts/data/processed/x_test.pkl'
    logger.info(f"Creating `{file_name}`")
    with open(file_name, 'wb') as handle:
        pickle.dump(x_test, handle)

    file_name = 'artifacts/data/processed/y_train.pkl'
    logger.info(f"Creating `{file_name}`")
    with open(file_name, 'wb') as handle:
        pickle.dump(y_train, handle)

    file_name = 'artifacts/data/processed/y_test.pkl'
    logger.info(f"Creating `{file_name}`")
    with open(file_name, 'wb') as handle:
        pickle.dump(y_test, handle)


if __name__ == '__main__':
    main()
