from palimpzest.constants import Cardinality
from palimpzest.corelib import File, Number, Schema
from palimpzest.elements import DataRecord
from typing import Any, Dict, List, Union

import os


class AbstractDataSource:
    """
    An AbstractDataSource is an Iterable which yields DataRecords adhering to a given schema.

    This base class must have its `__iter__` method implemented by a subclass, with each
    subclass reading data files from some real-world source (i.e. a directory, an S3 prefix,
    etc.).

    Many (if not all) DataSources should use Schemas from `palimpzest.elements.core`.
    In the future, programmers can implement their own DataSources using custom Schemas.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    def serialize(self) -> Dict[str, Any]:
        return {"schema": self.schema.jsonSchema()}

    def schema(self) -> Schema:
        return self.schema

    def getSize(self) -> int:
        raise NotImplementedError(f"You are calling this method from an abstract class.")

    def getItem(self, idx: int) -> DataRecord:
        raise NotImplementedError(f"You are calling this method from an abstract class.")


class DataSource(AbstractDataSource):
    def __init__(self, schema: Schema, dataset_id: str, cardinality: Cardinality = Cardinality.ONE_TO_ONE) -> None:
        super().__init__(schema)
        self.dataset_id = dataset_id
        self.cardinality = cardinality

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        return self.dataset_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}, dataset_id={self.dataset_id})"


class MemorySource(DataSource):
    """MemorySource returns multiple objects that reflect contents of an in-memory Python list"""

    def __init__(self, vals: List[Union[int, float]], dataset_id: str):
        # For the moment we assume that we are given a list of floats or ints, but someday it could be strings or something else
        super().__init__(Number, dataset_id)
        self.vals = vals

    def getSize(self):
        return len(self.vals)

    def getItem(self, idx: int):
        value = self.vals[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.value = value

        return dr


class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""

    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]
        self.path=path

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "directory",
        }

    # Consider making this the getSize method?
    def getMemorySize(self):
        # Get the memory size of the files in the directory
        return sum([os.path.getsize(filepath) for filepath in self.filepaths])

    # Consider making this the __len__ method?
    def getSize(self):
        return len(self.filepaths)

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = filepath
        with open(filepath, "rb") as f:
            dr.contents = f.read()

        return dr


class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""

    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.filepath = path

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "file",
        }

    def getSize(self):
        return 1

    def getItem(self, idx: int):
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = self.filepath
        with open(self.filepath, "rb") as f:
            dr.contents = f.read()

        return dr


class UserSource(DataSource):
    """UserSource is a DataSource that is created by the user and not loaded from a file"""

    def __init__(self, schema: Schema, dataset_id: str, cardinality: Cardinality = Cardinality.ONE_TO_ONE) -> None:
        super().__init__(schema, dataset_id, cardinality)

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "source_type": "user-defined:" + self.__class__.__name__,
        }

    def getSize(self):
        raise NotImplementedError("User needs to implement this method.")

    def getItem(self):
        raise NotImplementedError("User needs to implement this method.")
