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
import asyncio
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


def save_to_json(data: dict, file_name: str) -> None:
    with open(file_name, 'w') as f:
        # json.dump() method writes the JSON string directly to the file object, so we
        # don't have to create the JSON string first.
        json.dump(data, f)


class CorpusDataLoader(DataPersistence):
    """
    Class that saves and loads a Corpus to/from a set json files (one json per document).

    A directory is created that corresponds to the name of the object and the json files are
    stored in that directory.
    """
    def __init__(
            self,
            description: str,
            dependencies: list,
            directory: str,
            corpus_creator: Callable[[], Corpus],
            batch_size: int = 1000,
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
            cache=cache
        )
        self.directory = directory
        self._corpus_creator = corpus_creator
        self._batch_size = batch_size
        self.loop = asyncio.new_event_loop()

    @property
    def sub_directory(self) -> str:
        return os.path.join(self.directory, self.name + '__json')

    def _get_file_name(self, index) -> str:
        """Full path (directory and file name) to load/save."""
        return os.path.join(self.sub_directory, f'doc_{index}.json')

    def _read_json_files(self):
        """Generator that reads in the json docs one at a time."""
        doc_list = os.listdir(self.sub_directory)
        # sort files based on the doc index e.g. `doc_0.json`, `doc_1.json`
        # ensuring that the same order of the docs is maintained
        # sort values based on index, we can't rely on string sorting because the files might be
        # sorted like this:
        # file_1
        # file_10
        # file_11
        # ...
        # file_2
        # file_20
        doc_list = sorted(doc_list, key=lambda x: int(x.split('_')[1].split('.')[0]))
        for file_name in doc_list:
            with open(os.path.join(self.sub_directory, file_name), 'r') as f:
                yield json.load(f)

    async def _save_async(self, doc_dicts: list[dict], start_index):
        tasks = []
        for index, doc_dict in enumerate(doc_dicts):
            file_name = self._get_file_name(index=index + start_index)
            task = self.loop.run_in_executor(None, save_to_json, doc_dict, file_name)
            tasks.append(task)
        await asyncio.gather(*tasks)

    def _save(self, data: Corpus):
        logging.info(f"Saving Corpus object to `{self.sub_directory}`")
        # check if the directory exists; if it doesn't, create it; if it does, remove all files
        import shutil
        if os.path.exists(self.sub_directory):
            shutil.rmtree(self.sub_directory)
        os.mkdir(self.sub_directory)

        from itertools import islice
        gen = data.to_doc_dicts()
        start_index = 0
        for batch in iter(lambda: list(islice(gen, self._batch_size)), []):
            logging.info(f"Saving to JSON; processing {len(batch)} items; start_index = {start_index}...")  # noqa
            self.loop.run_until_complete(self._save_async(batch, start_index))
            start_index += len(batch)

    def _load(self) -> list[dict]:
        corpus = self._corpus_creator()
        corpus.from_doc_dicts(self._read_json_files())
        return corpus


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
            cache=False,
        )
        self.reddit_corpus = CorpusDataLoader(
            description="reddit dataset transformed to a corpus dataset",
            dependencies=['reddit'],
            directory='/code/artifacts/data/processed',
            corpus_creator=create_reddit_corpus_object,
            cache=False,
        )
        self.un_debates = PickledDataLoader(
            description="This dataset was copied from https://github.com/blueprints-for-text-analytics-python/blueprints-text/tree/master/data/un-general-debates",  # noqa
            dependencies=[],
            directory='/code/artifacts/data/raw',
            cache=False,
        )
        self.un_debate_paragraphs = PickledDataLoader(
            description="un_debates dataset transformed to paragraphs.",
            dependencies=['un_debates'],
            directory='/code/artifacts/data/processed',
            cache=False,
        )
        self.un_debate_corpus = CorpusDataLoader(
            description="un_debates_paragraphs transformed to a corpus dataset",
            dependencies=['un_debate_paragraphs'],
            directory='/code/artifacts/data/processed',
            corpus_creator=create_un_corpus_object,
            cache=False,
        )
        # call __init__() after defining properties
        super().__init__()


# create a global object that can be imported into other scripts
DATA = Datasets()

# ensure all names got set properly
assert all([getattr(DATA, x).name == x for x in DATA.datasets])
