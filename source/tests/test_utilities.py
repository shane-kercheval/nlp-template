import time
import unittest
import pandas as pd
import source.executables.helpers.classification_search_space as css
from source.executables.helpers.utilities import Timer, get_logger
from source.tests.helpers import get_test_file_path


class TestUtilities(unittest.TestCase):

    @staticmethod
    def to_string(obj):
        return str(obj). \
            replace(", '", ",\n'"). \
            replace('{', '{\n'). \
            replace('}', '\n}'). \
            replace(', ({', ',\n({')

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__timer(self):
        print('\n')
        with Timer("testing timer without logger"):
            time.sleep(0.05)

        with Timer("testing timer with logger", get_logger()):
            time.sleep(0.2)

        with Timer("testing another timer", get_logger()) as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer._interval)
