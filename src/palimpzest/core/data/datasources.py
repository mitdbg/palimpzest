from __future__ import annotations

import abc
import base64
import json
import os
import sys
from io import BytesIO
from typing import Any, Callable

import modal
import pandas as pd
from bs4 import BeautifulSoup
from papermage import Document

from palimpzest import constants
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import (
    DefaultSchema,
    File,
    ImageFile,
    PDFFile,
    Schema,
    TextFile,
    WebPage,
    XLSFile,
)
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

    def __init__(self, schema: Schema) -> None:
        self._schema = schema

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def get_item(self, idx: int) -> DataRecord: ...

    @abc.abstractmethod
    def get_size(self) -> int: ...

    @property
    def schema(self) -> Schema:
        return self._schema

    def copy(self) -> AbstractDataSource:
        raise NotImplementedError("You are calling this method from an abstract class.")

    def serialize(self) -> dict[str, Any]:
        return {"schema": self._schema.json_schema()}


class DataSource(AbstractDataSource):
    def __init__(self, schema: Schema, dataset_id: str) -> None:
        super().__init__(schema)
        self.dataset_id = dataset_id

    def universal_identifier(self):
        """Return a unique identifier for this Set."""
        return self.dataset_id

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}, dataset_id={self.dataset_id})"


# Second level of abstraction
class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""

    def __init__(self, path: str, dataset_id: str, schema: Schema) -> None:
        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]
        self.path = path
        super().__init__(schema, dataset_id)

    def serialize(self) -> dict[str, Any]:
        return {
            "schema": self.schema.json_schema(),
            "path": self.path,
            "source_type": "directory",
        }

    def __len__(self):
        return len(self.filepaths)

    def get_size(self):
        # Get the memory size of the files in the directory
        return sum([os.path.getsize(filepath) for filepath in self.filepaths])

    def get_item(self, idx: int):
        raise NotImplementedError("You are calling this method from an abstract class.")


class FileSource(DataSource):
    """FileSource returns a single File object from a single real-world local file"""

    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(File, dataset_id)
        self.filepath = path

    def copy(self):
        return FileSource(self.filepath, self.dataset_id)

    def serialize(self) -> dict[str, Any]:
        return {
            "schema": self.schema.json_schema(),
            "path": self.filepath,
            "source_type": "file",
        }

    def __len__(self):
        return 1

    def get_size(self):
        # Get the memory size of the filepath
        return os.path.getsize(self.filepath)

    def get_item(self, idx: int) -> DataRecord:
        dr = DataRecord(self.schema, source_id=self.filepath)
        dr.filename = self.filepath
        with open(self.filepath, "rb") as f:
            dr.contents = f.read()

        return dr


class MemorySource(DataSource):
    """MemorySource returns multiple objects that reflect contents of an in-memory Python list
        TODO(gerardo): Add support for other types of in-memory data structures (he has some code
                   for subclassing MemorySource on his branch)
    """

    def __init__(self, vals: Any, dataset_id: str = "default_memory_input"):
        if isinstance(vals, (str, int, float)):
            self.vals = [vals]
        elif isinstance(vals, tuple):
            self.vals = list(vals)
        else:
            self.vals = vals
        schema = Schema.from_df(self.vals) if isinstance(self.vals, pd.DataFrame) else DefaultSchema
        super().__init__(schema, dataset_id)

    def copy(self):
        return MemorySource(self.vals, self.dataset_id)

    def __len__(self):
        return len(self.vals)

    def get_size(self):
        return sum([sys.getsizeof(self.get_item(idx)) for idx in range(len(self))])

    def get_item(self, idx: int) -> DataRecord:
        dr = DataRecord(self.schema, source_id=idx)
        if isinstance(self.vals, pd.DataFrame):
            row = self.vals.iloc[idx]
            for field_name in row.index:
                field_name_str = f"column_{field_name}" if isinstance(field_name, (int, float)) else str(field_name)
                setattr(dr, field_name_str, row[field_name])
        else:
            dr.value = self.vals[idx]

        return dr


# Third level of abstraction
class HTMLFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=WebPage)
        assert all([filename.endswith(tuple(constants.HTML_EXTENSIONS)) for filename in self.filepaths])

    def copy(self):
        return HTMLFileDirectorySource(self.path, self.dataset_id)

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

    def get_item(self, idx: int) -> DataRecord:
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = os.path.basename(filepath)
        with open(filepath) as f:
            text_content = f.read()

        html = text_content
        tokens = html.split()[: constants.MAX_HTML_ROWS]
        dr.html = " ".join(tokens)

        stripped_html = self.html_to_text_with_links(text_content)
        tokens = stripped_html.split()[: constants.MAX_HTML_ROWS]
        dr.text = " ".join(tokens)

        return dr


class ImageFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=ImageFile)
        assert all([filename.endswith(tuple(constants.IMAGE_EXTENSIONS)) for filename in self.filepaths])

    def copy(self):
        return ImageFileDirectorySource(self.path, self.dataset_id)

    def get_item(self, idx: int) -> DataRecord:
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            dr.contents = base64.b64encode(f.read())
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

    def copy(self):
        return PDFFileDirectorySource(self.path, self.dataset_id)

    def get_item(self, idx: int) -> DataRecord:
        filepath = self.filepaths[idx]
        pdf_filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            pdf_bytes = f.read()

        if self.pdfprocessor == "modal":
            print("handling PDF processing remotely")
            remote_func = modal.Function.lookup("palimpzest.tools", "processPapermagePdf")
        else:
            remote_func = None

        # generate text_content from PDF
        if remote_func is not None:
            doc_json_str = remote_func.remote([pdf_bytes])
            docdict = json.loads(doc_json_str[0])
            doc = Document.from_json(docdict)
            text_content = ""
            for p in doc.pages:
                text_content += p.text
        else:
            text_content = get_text_from_pdf(pdf_filename, pdf_bytes, pdfprocessor=self.pdfprocessor, file_cache_dir=self.file_cache_dir)

        # construct data record
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = pdf_filename
        dr.contents = pdf_bytes
        dr.text_contents = text_content

        return dr


class TextFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=TextFile)

    def copy(self):
        return TextFileDirectorySource(self.path, self.dataset_id)

    def get_item(self, idx: int) -> DataRecord:
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = os.path.basename(filepath)
        with open(filepath) as f:
            dr.contents = f.read()
        return dr


class XLSFileDirectorySource(DirectorySource):
    def __init__(self, path: str, dataset_id: str) -> None:
        super().__init__(path=path, dataset_id=dataset_id, schema=XLSFile)
        assert all([filename.endswith(tuple(constants.XLS_EXTENSIONS)) for filename in self.filepaths])

    def copy(self):
        return XLSFileDirectorySource(self.path, self.dataset_id)

    def get_item(self, idx: int) -> DataRecord:
        filepath = self.filepaths[idx]
        dr = DataRecord(self.schema, source_id=filepath)
        dr.filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            dr.contents = f.read()

        xls = pd.ExcelFile(BytesIO(dr.contents), engine="openpyxl")
        dr.number_sheets = len(xls.sheet_names)
        dr.sheet_names = xls.sheet_names
        return dr


# User-defined datasources
class UserSource(DataSource):
    """UserSource is a DataSource that is created by the user and not loaded from a file"""

    def __init__(self, schema: Schema, dataset_id: str) -> None:
        super().__init__(schema, dataset_id)

    def serialize(self) -> dict[str, Any]:
        return {
            "schema": self.schema.json_schema(),
            "source_type": "user-defined:" + self.__class__.__name__,
        }

    def __len__(self):
        raise NotImplementedError("User needs to implement this method")

    def get_size(self):
        raise NotImplementedError("User may optionally implement this method.")

    def get_item(self, idx: int) -> DataRecord:
        raise NotImplementedError("User needs to implement this method.")

    def copy(self):
        raise NotImplementedError("User needs to implement this method.")

class ValidationDataSource(UserSource):
    """
    TODO: update this class interface (and comment)
    """

    def get_val_length(self) -> int:
        raise NotImplementedError("User needs to implement this method.")

    def get_field_to_metric_fn(self) -> Callable:
        raise NotImplementedError("User needs to implement this method.")

    def get_item(self, idx: int, val: bool = False, include_label: bool = False) -> DataRecord:
        raise NotImplementedError("User needs to implement this method.")
