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
import os
import datetime
import logging
import pickle
from abc import ABC, abstractmethod

import pandas as pd


class DataPersistence(ABC):
    """
    Class that wraps the logic of saving/loading/describing a given dataset.
    Meant to be subclassed with specific types of loaders (e.g. pickle, csv, database, etc.)
    """
    def __init__(self, description: str, dependencies: list):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
        """
        self.description = description
        self.dependencies = dependencies
        self.name = None  # this is set dynamically

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _save(self, data):
        pass

    def load(self):
        assert self.name
        return self._load()

    def save(self, data):
        assert self.name
        self._save(data)


class FileDataPersistence(DataPersistence):
    """
    Class that wraps the logic of saving/loading/describing a given dataset to the file-system.
    Adds logic for backing up datasets if they are being saved and already exist (i.e. renaming
    the file with a timestamp)
    Meant to be subclassed with specific types of loaders (e.g. pickle, csv, etc.)
    """
    def __init__(self, description: str, dependencies: list, directory: str):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
        """
        super().__init__(description, dependencies)
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
        return self._load()

    def save(self, data):
        assert self.name
        logging.info(f"Saving data `{self.name}` to `{self.path}`")
        # if the file already exists, save it to another name
        if os.path.isfile(self.path):
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            new_name = self.path + '.' + timestamp
            logging.info(f"Backing up current data `{self.name}` to `{new_name}`")
            os.rename(self.path, new_name)
        self._save(data)


class PickledDataLoader(FileDataPersistence):
    """
    Class that wraps the logic of saving/loading/describing a given dataset.
    """
    def __init__(self, description: str, dependencies: list, directory: str):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
            directory:
                the directory to save to and load from. NOTE: this should **not** contain the file
                name which is assigned at a later point in time based on the property name in the
                `Datasets` class.
        """
        super().__init__(description, dependencies, directory)

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
    def __init__(self, description: str, dependencies: list, directory: str):
        """
        Args:
            description: description of the dataset
            dependencies: dependencies of the dataset
            directory:
                the path to save to and load from. NOTE: this should **not** contain the file name
                which is assigned at a later point in time based on the property name in the
                `Datasets` class.
        """
        super().__init__(description, dependencies, directory)

    @property
    def file_extension(self):
        return '.csv'

    def _load(self):
        return pd.read_csv(self.path)

    def _save(self, data: pd.DataFrame):
        data.to_csv(self.path, index=None)


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
        self.dataset_1 = PickledDataLoader(
            description="Dataset description",
            dependencies=['SNOWFLAKE.SCHEMA.TABLE'],
            directory='/code/artifacts/data/raw',
        )
        self.other_dataset_2 = PickledDataLoader(
            description="Other dataset description",
            dependencies=['dataset_1'],
            directory='/code/artifacts/data/raw',
        )
        self.dataset_3_csv = CsvDataLoader(
            description="Other dataset description",
            dependencies=['dataset_1'],
            directory='/code/artifacts/data/raw',
        )
        # call __init__() after defining properties
        super().__init__()


# create a global object that can be imported into other scripts
DATA = Datasets()

# ensure all names got set properly
assert all([getattr(DATA, x).name == x for x in DATA.datasets])
