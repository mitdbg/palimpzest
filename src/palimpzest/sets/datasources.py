from palimpzest.elements import DataRecord, File, Schema
from typing import Any, Callable, Dict

import hashlib
import json
import os


class DataSource:
    """
    A DataSource is an Iterable which yields DataRecords adhering to a given schema.
    
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
    
    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hashlib.sha256(ordered.encode()).hexdigest()
        return result


class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""
    def __init__(self, path: str) -> None:
        super().__init__(File)
        self.path = path

    def __iter__(self) -> Callable[[], DataRecord]:
        def filteredIterator():
            for filename in os.listdir(self.path):
                file_path = os.path.join(self.path, filename)
                if os.path.isfile(file_path):
                    dr = DataRecord(self.schema)
                    dr.filename = file_path
                    bytes_data = open(file_path, "rb").read()
                    dr.contents = bytes_data
                    yield dr

        return filteredIterator()

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "directory",
        }


class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""
    def __init__(self, path: str) -> None:
        super().__init__(File)
        self.path = path

    def __iter__(self) -> Callable[[], DataRecord]:
        def filteredIterator():
            for path in [self.path]:
                dr = DataRecord(self.schema)
                dr.filename = path
                bytes_data = open(path, "rb").read()
                dr.contents = bytes_data

                yield dr

        return filteredIterator()

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "file",
        }
