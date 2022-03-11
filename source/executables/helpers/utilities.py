import datetime
import logging
import logging.config

import xmltodict
import yaml


def get_logger(config="source/config/logging/local.conf", logger_name='app', leg_level="DEBUG"):
    logging.config.fileConfig(config, disable_existing_loggers=False)
    logger = logging.getLogger(logger_name)
    logger.setLevel(leg_level)

    return logger


def open_dict_like_file(file_name):
    with open(file_name, "r") as f:
        if file_name.endswith("json"):
            result = yaml.load(f)
        elif file_name.endswith("yaml") or file_name.endswith("yml"):
            result = yaml.load(f)
        elif file_name.endswith("xml"):
            result = xmltodict.parse(f.read())
        else:
            logging.warning("%s not a known dictionary-like file type", file_name)
    return result


class Timer:
    def __init__(self, message, logger=None):
        self._logger = logger
        self._message = message

    def __enter__(self):
        if self._logger is None:
            print(f'Started: {self._message}')
        else:
            self._logger.info(f'Started: {self._message}')
        self._start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = f"Finished ({self._interval.total_seconds():.2f} seconds)"

        if self._logger is None:
            print(message)
        else:
            self._logger.info(message)
