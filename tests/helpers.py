import os

import pandas as pd
from helpsk.pandas import print_dataframe
from helpsk.utility import is_debugging, redirect_stdout_to_file


def get_test_file_path(file_path) -> str:
    """
    Returns the path to /tests folder, adjusting for the difference in the current working
    directory when debugging vs running from command line.
    """
    return os.path.join(os.getcwd(), 'tests/test_files', file_path)


def dataframe_to_text_file(dataframe, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_columns', 20)

    with redirect_stdout_to_file(file_name):
        print_dataframe(dataframe)
    assert os.path.isfile(file_name)
