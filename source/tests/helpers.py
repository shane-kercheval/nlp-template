import os
from helpsk.utility import is_debugging


def get_test_file_path(file_path) -> str:
    """Returns the path to /tests folder, adjusting for the difference in the current working directory when
    debugging vs running from command line.
    """
    path = os.getcwd()
    if not is_debugging():
        path = os.path.join(path, 'source/tests')

    path = os.path.join(path, 'test_files', file_path)
    return path
