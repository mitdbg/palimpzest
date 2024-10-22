import abc
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Type, Union

import modal
import pandas as pd
from bs4 import BeautifulSoup
from papermage import Document

from palimpzest import constants
from palimpzest.constants import Cardinality
from palimpzest.corelib import File, Number, Schema
from palimpzest.corelib.schemas import ImageFile, PDFFile, TextFile, WebPage, XLSFile
from palimpzest.elements import DataRecord
from palimpzest.tools.pdfparser import get_text_from_pdf


# First level of abstraction
class AbstractDataSource(abc.ABC):
    """
    An AbstractDataSource is an Iterable which yields DataRecords adhering to a given schema.

    This base class must have its `__iter__` method implemented by a subclass, with each
    subclass reading data files from some real-world source (i.e. a directory, an S3 prefix,
    etc.).

    Many (if not all) DataSources should use Schemas from `palimpzest.elements.core`.
    In the future, programmers can implement their own DataSources using custom Schemas.
    """

    def __init__(self, schema: Type[Schema]) -> None:
        self._schema = schema

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def getItem(self, idx: int) -> DataRecord: ...

    @abc.abstractmethod
    def getSize(self) -> int: ...

    @property
    def schema(self) -> Type[Schema]:
        return self._schema

    def serialize(self) -> Dict[str, Any]:
        return {"schema": self._schema.jsonSchema()}


class DataSource(AbstractDataSource):
    def __init__(
        self, schema: Type[Schema], dataset_id: str, cardinality: Cardinality = Cardinality.ONE_TO_ONE
    ) -> None:
        super().__init__(schema)
        self.dataset_id = dataset_id
        self.cardinality = cardinality

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        return self.dataset_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}, dataset_id={self.dataset_id})"


# Second level of abstraction
class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""

    def __init__(self, path: str, dataset_id: str, schema: Type[Schema]) -> None:
        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]
        self.path = path
        super().__init__(schema, dataset_id)

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.path,
            "source_type": "directory",
        }

    def __len__(self):
        return len(self.filepaths)

    def getSize(self):
        # Get the memory size of the files in the directory
        return sum([os.path.getsize(filepath) for filepath in self.filepaths])

    def getItem(self, idx: int):
        raise NotImplementedError("You are calling this method from an abstract class.")


class MemorySource(DataSource):
    """MemorySource returns multiple objects that reflect contents of an in-memory Python list"""

    def __init__(self, vals: List[Union[int, float]], dataset_id: str):
        # For the moment we assume that we are given a list of floats or ints, but someday it could be strings or something else
        super().__init__(Number, dataset_id)
        self.vals = vals

    def __len__(self):
        return len(self.vals)

    def getSize(self):
        return sum([sys.getsizeof(self.getItem(idx)) for idx in range(len(self))])

    def getItem(self, idx: int):
        value = self.vals[idx]
        dr = DataRecord(self._schema, scan_idx=idx)
        dr.value = value

        return dr


class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""

    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.filepath = path

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "path": self.filepath,
            "source_type": "file",
        }

    def __len__(self):
        return 1

    def getSize(self):
        # Get the memory size of the filepath
        return os.path.getsize(self.filepath)

    def getItem(self, idx: int):
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = self.filepath
        with open(self.filepath, "rb") as f:
            dr.contents = f.read()

        return dr


class UserSource(DataSource):
    """UserSource is a DataSource that is created by the user and not loaded from a file"""

    def __init__(
        self, schema: Type[Schema], dataset_id: str, cardinality: Cardinality = Cardinality.ONE_TO_ONE
    ) -> None:
        super().__init__(schema, dataset_id, cardinality)

    def serialize(self) -> Dict[str, Any]:
        return {
            "schema": self.schema.jsonSchema(),
            "source_type": "user-defined:" + self.__class__.__name__,
        }

    def __len__(self):
        raise NotImplementedError("User needs to implement this method")

    def getSize(self):
        raise NotImplementedError("User may optionally implement this method.")

    def getItem(self, idx: int):
        raise NotImplementedError("User needs to implement this method.")


# Third level of abstraction
class HTMLFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=WebPage)
        assert all([filename.endswith(tuple(constants.HTML_EXTENSIONS)) for filename in self.filepaths])

    def html_to_text_with_links(self, html):
        # Parse the HTML content
        soup = BeautifulSoup(html, "html.parser")

        # Find all hyperlink tags
        for a in soup.find_all("a"):
            # Check if the hyperlink tag has an 'href' attribute
            if a.has_attr("href"):
                # Replace the hyperlink with its text and URL in parentheses
                a.replace_with(f"{a.text} ({a['href']})")

        # Extract text from the modified HTML
        text = soup.get_text(separator="\n", strip=True)
        return text

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            textcontent = f.read()

        html = textcontent
        tokens = html.split()[:constants.MAX_HTML_ROWS]
        dr.html = " ".join(tokens)

        strippedHtml = self.html_to_text_with_links(textcontent)
        tokens = strippedHtml.split()[:constants.MAX_HTML_ROWS]
        dr.text = " ".join(tokens)

        return dr


class ImageFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=ImageFile())
        assert all([filename.endswith(tuple(constants.IMAGE_EXTENSIONS)) for filename in self.filepaths])

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            dr.contents = f.read()
        return dr


class PDFFileDirectorySource(DirectorySource):
    def __init__(
        self,
        path: str,
        dataset_id: str,
        pdfprocessor: str = "modal",
        file_cache_dir: str = "/tmp",
    ) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=PDFFile)
        assert all([filename.endswith(tuple(constants.PDF_EXTENSIONS)) for filename in self.filepaths])
        self.pdfprocessor = pdfprocessor
        self.file_cache_dir = file_cache_dir

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        pdf_filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()

        if self.pdfprocessor == "modal":
            print("handling PDF processing remotely")
            remoteFunc = modal.Function.lookup("palimpzest.tools", "processPapermagePdf")
        else:
            remoteFunc = None

        # generate text_content from PDF
        if remoteFunc is not None:
            docJsonStr = remoteFunc.remote([pdf_bytes])
            docdict = json.loads(docJsonStr[0])
            doc = Document.from_json(docdict)
            text_content = ""
            for p in doc.pages:
                text_content += p.text
        else:
            text_content = get_text_from_pdf(pdf_filename, pdf_bytes, file_cache_dir=self.file_cache_dir)

        # construct data record
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = pdf_filename
        dr.contents = pdf_bytes
        dr.text_contents = text_content[:15000]  # TODO Very hacky

        return dr


class TextFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=TextFile)

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "r") as f:
            dr.contents = f.read()
        return dr


class XLSFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=XLSFile)
        assert all([filename.endswith(tuple(constants.XLS_EXTENSIONS)) for filename in self.filepaths])

    def getItem(self, idx: int):
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, scan_idx=idx)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            dr.contents = f.read()

        xls = pd.ExcelFile(BytesIO(dr.contents), engine="openpyxl")
        dr.number_sheets = len(xls.sheet_names)
        dr.sheet_names = xls.sheet_names
        return dr
