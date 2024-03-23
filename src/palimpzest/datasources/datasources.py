from palimpzest.elements import DataRecord, File, Number, Schema, RawJSONObject
from typing import Any, Callable, Dict, List, Union

import os
import time
import requests
import json


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

class DataSource(AbstractDataSource):
    def __init__(self, schema: Schema, dataset_id: str) -> None:
        super().__init__(schema)
        self.dataset_id = dataset_id

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

    def __iter__(self) -> Callable[[], DataRecord]:
        def valIterator():
            for v in self.vals:
                dr = DataRecord(self.schema)
                dr.value = v
                yield dr

        return valIterator()


class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
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
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
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

class StreamingJSONSource(DataSource):
    """StreamingSource returns multiple objects from a stream of data"""
    def __init__(self, url:str, blockTime: int, dataset_id: str):
        super().__init__(RawJSONObject, dataset_id)
        self.url = url
        self.blockTime = blockTime

    def __iter__(self) -> Callable[[], DataRecord]:
        def streamIterator():
            seenItems = set()
            timeLastNewItemSeen = time.time()
            lastTry = -1

            while True:
                if self.blockTime >= 0 and time.time() - timeLastNewItemSeen > self.blockTime:
                    break

                if lastTry > 0 and time.time() - lastTry < 5:
                    time.sleep(5)
                response = requests.get(self.url, timeout=5)
                lastTry = time.time()
                if response.status_code == 200:
                    results = response.json()
                    for result in results:
                        # convert json object to string
                        resultStr = json.dumps(result)
                        if resultStr in seenItems:
                            continue

                        dr = DataRecord(self.schema)
                        dr.json = resultStr
                        seenItems.add(resultStr)
                        yield dr
                        timeLastNewItemSeen = time.time()

        return streamIterator()

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "url": self.url,
            "blockTime": self.blockTime,
            "source_type": "jsonstream",
        }