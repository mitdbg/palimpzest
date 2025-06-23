import os
from abc import ABC, abstractmethod

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import Schema
from palimpzest.query.operators.logical import ContextScan


class Context(Dataset, ABC):
    """
    The `Context` class is an abstract base class for root `Datasets` whose data is accessed
    via user-defined methods. Classes which inherit from this class must implement two methods:

    - `list_filepaths()`: which lists the files that the `Context` has access to.
    - `read_filepath(path: str)`: which reads the file corresponding to the given `path`.
    
    A `Context` is a special type of `Dataset` that represents a view over an underlying `Dataset`.
    Each `Context` has a `name` which uniquely identifies it, as well as a natural language `description`
    of the data / computation that the `Context` represents. Similar to `Dataset`s, `Context`s can be
    lazily transformed using functions such as `sem_filter`, `sem_map`, `sem_join`, etc., and they may
    be materialized or unmaterialized.
    """

    def __init__(self, id: str, description: str) -> None:
        """
            Constructor for the `Context` class.

            Args:
                id (str): a string identifier for the `Context`
                description (str): The description of the data contained within the `Context`
        """
        # set the id for the Dataset
        self._id = id

        # set the description
        self._description = description

        # compute Schema and call parent constructor
        schema = Schema.from_json([{"id": self._id, "desc": "The name of the context", "type": str}])
        super().__init__(sources=None, operator=ContextScan(datasource=self, output_schema=schema), schema=schema)

    @abstractmethod
    def list_filepaths(self) -> list[str]:
        """
        Returns the list of files which the `Context` has access to.
        """
        pass

    @abstractmethod
    def read_filepath(self, path: str) -> str:
        """
        Reads the file at the given `path` and returns its contents.
        """
        pass


class TextFileContext(Context):
    def __init__(self, path: str, **kwargs) -> None:
        """
        Constructor for the `TextFileContext` class.

        Args:
            path (str): The path to the file
            kwargs (dict): Keyword arguments containing the `Context's` id and description.
        """
        # check that path is a valid file or directory
        assert os.path.isfile(path) or os.path.isdir(path), f"Path {path} is not a file nor a directory"

        # get list of filepaths
        self.filepaths = []
        if os.path.isfile(path):
            self.filepaths = [path]
        else:
            self.filepaths = [
                os.path.join(path, filename)
                for filename in sorted(os.listdir(path))
                if os.path.isfile(os.path.join(path, filename))
            ]

        # call parent constructor to set id, operator, and schema
        super().__init__(**kwargs)

    def list_filepaths(self) -> list[str]:
        return self.filepaths

    def read_filepath(self, path: str) -> str:
        contents = None
        with open(path) as f:
            contents = f.read()
        
        return contents
