import pytest
import pandas as pd

from tests.helpers import get_test_file_path


@pytest.fixture
def reddit():
    return pd.read_pickle(get_test_file_path('datasets/reddit__sample.pkl'))
