"""
This file defines classes that hides the logic/path for saving and loading specific datasets that
are used across this project, as well as providing a brief description for each dataset.

To define a new dataset, create a property in Datasets.__init__() following the existing patthern.

The DATA variable is assigned an instance of the Datasets class and can be imported into other
scripts/notebooks.

To load the dataset called `the_dataset`, use the following code:

```
from source.services.data import DATA
df = DATA.the_dataset.load()
```

To save the dataset called `the_dataset`, use the following code:

```
from source.services.data import DATA

df = ...logic..
DATA.the_dataset.save(df)
```
"""
import json
import os
import datetime
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

from source.library.spacy import Corpus
from source.library.text_preparation import clean


class DataPersistence(ABC):
    """
    Class that wraps the logic of saving/loading/describing a given dataset.
    Meant to be subclassed with specific types of loaders (e.g. pickle, csv, database, etc.)
    """
    def __init__(self, description: str, dependencies: list, cache: bool = False):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
        """
        self.description = description
        self.dependencies = dependencies
        self.cache = cache
        self.name = None  # this is set dynamically
        self._cached_data = None

    def clear_cache(self):
        self._cached_data = None

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _save(self, data):
        pass

    def load(self):
        assert self.name
        if self.cache:
            if self._cached_data is None:
                self._cached_data = self._load()
            return self._cached_data
        else:
            return self._load()

    def save(self, data):
        assert self.name
        if self.cache:
            self._cached_data = data
        self._save(data)


class FileDataPersistence(DataPersistence):
    """
    Class that wraps the logic of saving/loading/describing a given dataset to the file-system.
    Adds logic for backing up datasets if they are being saved and already exist (i.e. renaming
    the file with a timestamp)
    Meant to be subclassed with specific types of loaders (e.g. pickle, csv, etc.)
    """
    def __init__(self, description: str, dependencies: list, directory: str, cache: bool = False):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
        """
        super().__init__(description=description, dependencies=dependencies, cache=cache)
        self.directory = directory

    @abstractmethod
    def _load(self):
        """Logic to load the `data`"""
        pass

    @abstractmethod
    def _save(self, data):
        """Logic to save the `data`"""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension to use for the path (e.g. '.csv' or '.pkl')"""
        pass

    @property
    def path(self) -> str:
        """Full path (directory and file name) to load/save."""
        return os.path.join(self.directory, self.name + self.file_extension)

    def load(self):
        assert self.name
        logging.info(f"Loading data `{self.name}` from `{self.path}`")
        return super().load()

    def save(self, data):
        assert self.name
        logging.info(f"Saving data `{self.name}` to `{self.path}`")
        # if the file already exists, save it to another name
        if os.path.isfile(self.path):
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            new_name = self.path + '.' + timestamp
            logging.info(f"Backing up current data `{self.name}` to `{new_name}`")
            os.rename(self.path, new_name)
        super().save(data)


class PickledDataLoader(FileDataPersistence):
    """
    Class that wraps the logic of saving/loading/describing a given dataset.
    """
    def __init__(self, description: str, dependencies: list, directory: str, cache: bool = False):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
            directory:
                the directory to save to and load from. NOTE: this should **not** contain the file
                name which is assigned at a later point in time based on the property name in the
                `Datasets` class.
        """
        super().__init__(
            description=description,
            dependencies=dependencies,
            directory=directory,
            cache=cache
        )

    @property
    def file_extension(self):
        return '.pkl'

    def _load(self):
        with open(self.path, 'rb') as handle:
            unpickled_object = pickle.load(handle)
        return unpickled_object

    def _save(self, data):
        with open(self.path, 'wb') as handle:
            pickle.dump(data, handle)


class CsvDataLoader(FileDataPersistence):
    """
    Class that wraps the logic of saving/loading/describing a given dataset.
    """
    def __init__(self, description: str, dependencies: list, directory: str, cache: bool = False):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
            directory:
                the path to save to and load from. NOTE: this should **not** contain the file name
                which is assigned at a later point in time based on the property name in the
                `Datasets` class.
        """
        super().__init__(
            description=description,
            dependencies=dependencies,
            directory=directory,
            cache=cache
        )

    @property
    def file_extension(self):
        return '.csv'

    def _load(self):
        return pd.read_csv(self.path)

    def _save(self, data: pd.DataFrame):
        data.to_csv(self.path, index=None)


def create_reddit_corpus_object() -> Corpus:
    """
    This function ensures we create the reddit Corpus object in the same way each time.
    We do not want to serialize the entire Corpus object because we initialize it with a function
    for cleaning new/unseen text, which we can't serialize to json.
    """
    stop_words_to_add = {'dear', 'regard', '_number_', '_tag_'}
    stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}
    corpus = Corpus(
        stop_words_to_add=stop_words_to_add,
        stop_words_to_remove=stop_words_to_remove,
        pre_process=clean,
        spacy_model='en_core_web_sm',
        sklearn_tokenenizer_min_df=20,
        sklearn_tokenenizer_max_tokens=200,
        sklearn_tokenenizer_max_bi_grams=30,
    )
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    return corpus


def create_un_corpus_object() -> Corpus:
    """
    This function ensures we create the reddit Corpus object in the same way each time.
    We do not want to serialize the entire Corpus object because we initialize it with a function
    for cleaning new/unseen text, which we can't serialize to json.
    """
    stop_words_to_add = {'dear', 'regard', '_number_', '_tag_'}
    stop_words_to_remove = {'down', 'no', 'none', 'nothing', 'keep'}
    corpus = Corpus(
        stop_words_to_add=stop_words_to_add,
        stop_words_to_remove=stop_words_to_remove,
        pre_process=clean,
        spacy_model='en_core_web_sm',
        sklearn_tokenenizer_min_df=20,
        sklearn_tokenenizer_max_tokens=200,
        sklearn_tokenenizer_max_bi_grams=30,
    )
    assert all(x in corpus.stop_words for x in stop_words_to_add)
    assert all(x not in corpus.stop_words for x in stop_words_to_remove)
    return corpus


class CorpusDataLoader(FileDataPersistence):
    """
    Class that saves and loads a Corpus to/from a json file.
    """
    def __init__(
            self,
            description: str,
            dependencies: list,
            directory: str,
            corpus_creator: Callable[[], Corpus],
            cache: bool = False):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
            corpus_creator: callable that creates a Corpus object
        """
        super().__init__(
            description=description,
            dependencies=dependencies,
            directory=directory,
            cache=cache
        )
        self._corpus_creator = corpus_creator

    @property
    def file_extension(self):
        return '.json'

    def _load(self) -> list[dict]:
        with open(self.path, 'r') as f:
            loaded_json = json.load(f)
        corpus = self._corpus_creator()
        corpus.from_doc_dicts(loaded_json)
        return corpus

    def _save(self, data: Corpus):
        with open(self.path, 'w') as f:
            json.dump(data.to_doc_dicts(), f)


class DatasetsBase(ABC):
    """
    class that defines all of the datasets available globally to the project.
    NOTE: in overridding the base class, call __init__() after defining properties
    """
    def __init__(self) -> None:
        """Use this function to define datasets by following the existing pattern."""
        # dynamically set the name property in the DataPersistence object in all of the object;
        # I don't love this design, but it forces the names to match the property name and reduces
        # the redundancy of duplicating the name when defining the property and passing in the name
        # ot the loader
        for dataset in self.datasets:
            dataset_obj = getattr(self, dataset)
            dataset_obj.name = dataset

    @property
    def datasets(self) -> list[str]:
        """Returns the names of the datasets available."""
        ignore = set(['datasets', 'descriptions', 'dependencies'])
        return [
            attr for attr in dir(self)
            if attr not in ignore and isinstance(getattr(self, attr), DataPersistence)
        ]

    @property
    def descriptions(self) -> dict[str]:
        """Returns the names and descriptions of the datasets available."""
        return [
            dict(
                dataset=x,
                description=getattr(self, x).description
            )
            for x in self.datasets
        ]

    @property
    def dependencies(self) -> dict[str]:
        """Returns the names and dependencies of the datasets available."""
        return [
            dict(
                dataset=x,
                dependencies=getattr(self, x).dependencies
            )
            for x in self.datasets
        ]


class Datasets(DatasetsBase):
    def __init__(self) -> None:
        # define the datasets before calling __init__()
        self.reddit = PickledDataLoader(
            description="This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/reddit-selfposts",  # noqa
            dependencies=[],
            directory='/code/artifacts/data/raw',
        )
        self.reddit_corpus = PickledDataLoader(
            description="reddit dataset transformed to a corpus dataset",
            dependencies=['reddit'],
            directory='/code/artifacts/data/processed',
        )
        self.un_debates = PickledDataLoader(
            description="This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/un-general-debates",  # noqa
            dependencies=[],
            directory='/code/artifacts/data/raw',
        )
        self.un_debate_paragraphs = PickledDataLoader(
            description="un_debates dataset transformed to paragraphs.",
            dependencies=['un_debates'],
            directory='/code/artifacts/data/processed',
        )
        self.un_debate_corpus = PickledDataLoader(
            description="un_debates_paragraphs transformed to a corpus dataset",
            dependencies=['un_debate_paragraphs'],
            directory='/code/artifacts/data/processed',
        )
        # call __init__() after defining properties
        super().__init__()


# create a global object that can be imported into other scripts
DATA = Datasets()

# ensure all names got set properly
assert all([getattr(DATA, x).name == x for x in DATA.datasets])
