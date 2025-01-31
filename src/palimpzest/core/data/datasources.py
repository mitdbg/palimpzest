from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import modal
import pandas as pd
from bs4 import BeautifulSoup
from papermage import Document

from palimpzest import constants
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
class DataSource(ABC):
    """
    The `DataSource` represents a source of data for the PZ system. Every PZ program will have
    at least one `DataSource` object, which will generate data that is processed by PZ.

    Subclasses of the (abstract) `DataSource` class must implement two methods:
    - `__len__`: which returns the number of elements in the data source
    - `get_item(idx)`: which takes in an `idx` and returns the element at that index
    """

    def __init__(self, schema: Schema, dataset_id: str) -> None:
        """
        Constructor for the `DataSource` class.

        Args:
            schema (Schema): The output schema of the data source
            dataset_id (str): The unique identifier for the dataset
        """
        self._schema = schema  # NOTE: _schema currently has to match attribute name in Dataset
        self.dataset_id = dataset_id

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema}, dataset_id={self.dataset_id})"
    
    @property
    def schema(self) -> Schema:
        return self._schema
    
    def universal_identifier(self) -> str:
        """
        Return a unique identifier for this `DataSource`.
        NOTE: this currently has to mirror the `Dataset`'s `universal_identifier` method.
        """
        return self.dataset_id

    def serialize(self) -> dict:
        return {"schema": self._schema.json_schema()}

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the data source."""
        pass

    @abstractmethod
    def get_item(self, idx: int) -> dict:
        """
        Returns a single item from the data source at the given index.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary representing the item at the given index. The dictionary
            keys (i.e. fields) should match the fields specified in the schema of the
            data source, and the values should be the values associated with those fields.

            .. code-block:: python

                # Example return value
                {"field1": value1, "field2": value2, ...}

            If the data source is a validation data source, then the dictionary should
            be modified as follows:

            1. Place the input fields under a top-level key called `"fields"`
            2. Include a top-level key called `"labels"` which contains the groundtruth values for each **output field**
               which should be computed by the PZ program.
            3. (Optional), include a top-level key called `"score_fn"` which contains the function(s) for
               scoring the computed values against the groundtruth values. If no scoring function is provided
               for a given field then PZ will use exact matching as the default scoring function.

            Scoring function(s) only need to be provided for fields that require non-exact matching. The scoring
            function accepts two arguments, `output` and `target`, and returns a float in [0, 1] (higher is better)
            The `output` will be filled in by the PZ program, and the `target` will be the groundtruth value provided
            for that field in the `"labels"` dictionary.

            Example return values are shown below:

            .. code-block:: python
            
                # Example return value for a validation data source with exact match scoring;
                # suppose PZ is asked to compute the "first_name" and "age" fields (in the year 2025)
                {
                    "fields": {"name": "Jane Doe", "birthday": "01/01/1990"},
                    "labels": {"first_name": "Jane", "age": 35},
                }

                # Example return value for a validation data source with custom scoring functions;
                # suppose PZ is asked to compute the "fruits" mentioned in the input document, with recall as the scoring function

                def compute_recall(output, target):
                    tp = 0
                    for fruit in target:
                        if fruit in output:
                            tp += 1
                    return tp / len(target)
                ...
                {
                    "fields": {"document": "I like apples, oranges, and bananas."},
                    "labels": {"fruits": ["apples", "oranges", "bananas"]},
                    "score_fn": {"fruits": compute_recall},
                }

                # Example return value for a validation data source with custom and exact match scoring functions;
                # suppose PZ is asked to compute the "first_name" and "fruits" metnioned in the input document
                def compute_recall(output, target):
                    # same definition as above
                ...
                {
                    "fields": {"document": "Jane Doe like apples, oranges, and bananas."},
                    "labels": {"first_name": "Jane", "fruits": ["apples", "oranges", "bananas"]},
                    "score_fn": {"fruits": compute_recall},
                }
        """
        pass


# Second level of abstraction
class DirectorySource(DataSource):
    """
    DirectorySource returns a dictionary for each file in a directory. Each dictionary contains the filename and
    contents of a single file in the directory.
    """

    def __init__(self, path: str, dataset_id: str, schema: Schema) -> None:
        """
        Constructor for the `DirectorySource` class.

        Args:
            path (str): The path to the directory
            dataset_id (str): The unique identifier for the dataset
            schema (Schema): The output schema of the data source
        """
        assert os.path.isdir(path), f"Path {path} is not a directory"

        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]
        self.path = path
        super().__init__(schema, dataset_id)

    def serialize(self) -> dict:
        return {
            "schema": self.schema.json_schema(),
            "path": self.path,
            "source_type": "directory",
        }

    def __len__(self) -> int:
        return len(self.filepaths)

    def get_item(self, idx: int) -> dict:
        raise NotImplementedError("You are calling this method from an abstract class.")


class FileSource(DataSource):
    """FileSource returns a single dictionary with the filename and contents of a local file (in bytes)."""

    def __init__(self, path: str, dataset_id: str) -> None:
        """
        Constructor for the `FileSource` class. The `schema` is set to the default `File` schema.

        Args:
            path (str): The path to the file
            dataset_id (str): The unique identifier for the dataset
        """
        super().__init__(File, dataset_id)
        self.filepath = path

    def serialize(self) -> dict:
        return {
            "schema": self.schema.json_schema(),
            "path": self.filepath,
            "source_type": "file",
        }

    def __len__(self) -> int:
        return 1

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename and contents of the file.

        Args:
            idx (int): The index of the item to return. This argument is ignored.
        
        Returns:
            dict: A dictionary with the filename and contents of the file.

            .. code-block:: python

                {
                    "filename": "path/to/file.txt",
                    "contents": b"file contents here",
                }
        """
        filename = self.filepath
        with open(self.filepath, "rb") as f:
            contents = f.read()

        return {"filename": filename, "contents": contents}


class MemorySource(DataSource):
    """
    MemorySource returns one or more dictionaries that reflect the contents of an in-memory Python object `vals`.
    If `vals` is not a pd.DataFrame, then the dictionary returned by `get_item()` has a single field called "value".
    Otherwise, the dictionary contains the key-value mapping from columns to values for the `idx` row in the dataframe.

    TODO(gerardo): Add support for other types of in-memory data structures (he has some code for subclassing
        MemorySource on his branch)
    """

    def __init__(self, vals: Any, dataset_id: str = "default_memory_input") -> None:
        """
        Constructor for the `MemorySource` class. The `schema` is set to the default `DefaultSchema` schema.
        If `vals` is a pd.DataFrame, then the schema is set to the schema inferred from the DataFrame.

        Args:
            vals (Any): The in-memory object to use as the data source
            dataset_id (str): The unique identifier for the dataset
        """
        if isinstance(vals, (str, int, float)):
            self.vals = [vals]
        elif isinstance(vals, tuple):
            self.vals = list(vals)
        else:
            self.vals = vals
        schema = Schema.from_df(self.vals) if isinstance(self.vals, pd.DataFrame) else DefaultSchema
        super().__init__(schema, dataset_id)

    def __len__(self) -> int:
        return len(self.vals)

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the value(s) for the element at the specified `idx` in `vals`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: If `vals` is not a pd.DataFrame, then the dictionary has a single field called "value".
                Otherwise, the dictionary contains the key-value mapping from columns to values for the
                `idx` row in the dataframe.
            
            .. code-block:: python
            
                # Example return value at idx = 0, for the following list of values
                # [42, 43, 44, ...]
                {"value": 42}

                # Example return value at idx = 0, for the following DataFrame:
                # +---------+---------+---------+
                # |  name   |   job   |  hobby  |
                # +---------+---------+---------+
                # |  Alice  |  doctor |  tennis |
                # |  Bob    |  lawyer |  chess  |
                # +---------+---------+---------+
                {"column_name": "Alice", "column_job": "doctor", "column_hobby": "tennis"}
        """
        item = {}
        if isinstance(self.vals, pd.DataFrame):
            row = self.vals.iloc[idx]
            for field_name in row.index:
                field_name_str = f"column_{field_name}" if isinstance(field_name, (int, float)) else str(field_name)
                item[field_name_str] =  row[field_name]
        else:
            item["value"] = self.vals[idx]

        return item


# Third level of abstraction
class HTMLFileDirectorySource(DirectorySource):
    """
    HTMLFileDirectorySource returns a dictionary for each HTML file in a directory. Each dictionary contains the
    filename, raw HTML content, and parsed content of a single HTML file in the directory.
    """
    def __init__(self, path: str, dataset_id: str) -> None:
        """
        Constructor for the `HTMLFileDirectorySource` class. The `schema` is set to the `WebPage` schema.

        Args:
            path (str): The path to the directory
            dataset_id (str): The unique identifier for the dataset
        """
        super().__init__(path=path, dataset_id=dataset_id, schema=WebPage)
        assert all([filename.endswith(tuple(constants.HTML_EXTENSIONS)) for filename in self.filepaths])

    def _html_to_text_with_links(self, html: str) -> str:
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

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename, raw HTML content, and parsed content of the HTML file at the
        specified `idx`.

        Args:
            idx (int): The index of the item to return
        
        Returns:
            dict: A dictionary with the filename, raw HTML content, and parsed content of the HTML file.

            .. code-block:: python

                {
                    "filename": "file.html",
                    "html": "raw HTML content here",
                    "text": "parsed text content here",
                }
        """
        item = {}
        filepath = self.filepaths[idx]
        item["filename"] = os.path.basename(filepath)
        with open(filepath) as f:
            text_content = f.read()

        html = text_content
        tokens = html.split()[: constants.MAX_HTML_ROWS]
        item["html"] = " ".join(tokens)

        stripped_html = self._html_to_text_with_links(text_content)
        tokens = stripped_html.split()[: constants.MAX_HTML_ROWS]
        item["text"] = " ".join(tokens)

        return item


class ImageFileDirectorySource(DirectorySource):
    """
    ImageFileDirectorySource returns a dictionary for each image file in a directory. Each dictionary contains the
    filename and the base64 encoded bytes content of a single image file in the directory.
    """
    def __init__(self, path: str, dataset_id: str) -> None:
        """
        Constructor for the `ImageFileDirectorySource` class. The `schema` is set to the `ImageFile` schema.

        Args:
            path (str): The path to the directory
            dataset_id (str): The unique identifier for the dataset
        """
        super().__init__(path=path, dataset_id=dataset_id, schema=ImageFile)
        assert all([filename.endswith(tuple(constants.IMAGE_EXTENSIONS)) for filename in self.filepaths])

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename and base64 encoded bytes content of the image file at the
        specified `idx`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary with the filename and base64 encoded bytes content of the image file.

            .. code-block:: python

                {
                    "filename": "image.jpg",
                    "contents": b"base64 encoded image content here",
                }
        """
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            contents = base64.b64encode(f.read())

        return {"filename": filename, "contents": contents}


class PDFFileDirectorySource(DirectorySource):
    """
    PDFFileDirectorySource returns a dictionary for each PDF file in a directory. Each dictionary contains the
    filename, raw PDF content, and parsed text content of a single PDF file in the directory.

    This class also uses one of a predefined set of PDF processors to extract text content from the PDF files.
    """
    def __init__(
        self,
        path: str,
        dataset_id: str,
        pdfprocessor: str = "modal",
        file_cache_dir: str = "/tmp",
    ) -> None:
        """
        Constructor for the `PDFFileDirectorySource` class. The `schema` is set to the `PDFFile` schema.

        Args:
            path (str): The path to the directory
            dataset_id (str): The unique identifier for the dataset
            pdfprocessor (str): The PDF processor to use for extracting text content from the PDF files
            file_cache_dir (str): The directory to store the temporary files generated during PDF processing
        """
        super().__init__(path=path, dataset_id=dataset_id, schema=PDFFile)
        assert all([filename.endswith(tuple(constants.PDF_EXTENSIONS)) for filename in self.filepaths])
        self.pdfprocessor = pdfprocessor
        self.file_cache_dir = file_cache_dir

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename, raw PDF content, and parsed text content of the PDF file at the
        specified `idx`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary with the filename, raw PDF content, and parsed text content of the PDF file.

            .. code-block:: python

                {
                    "filename": "file.pdf",
                    "contents": b"raw PDF content here",
                    "text_contents": "parsed text content here",
                }
        """
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

        # construct and return item
        return {"filename": pdf_filename, "contents": pdf_bytes, "text_contents": text_content}


class TextFileDirectorySource(DirectorySource):
    """
    TextFileDirectorySource returns a dictionary for each text file in a directory. Each dictionary contains the
    filename and contents of a single text file in the directory.
    """
    def __init__(self, path: str, dataset_id: str) -> None:
        """
        Constructor for the `TextFileDirectorySource` class. The `schema` is set to the `TextFile` schema.

        Args:
            path (str): The path to the directory
            dataset_id (str): The unique identifier for the dataset
        """
        super().__init__(path=path, dataset_id=dataset_id, schema=TextFile)

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename and contents of the text file at the specified `idx`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary with the filename and contents of the text file.

            .. code-block:: python

                {
                    "filename": "file.txt",
                    "contents": "text content here",
                }
        """
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            contents = f.read()

        return {"filename": filename, "contents": contents}


class XLSFileDirectorySource(DirectorySource):
    """
    XLSFileDirectorySource returns a dictionary for each XLS file in a directory. Each dictionary contains the
    filename, contents, sheet names, and the number of sheets for a single XLS file in the directory.
    """
    def __init__(self, path: str, dataset_id: str) -> None:
        """
        Constructor for the `XLSFileDirectorySource` class. The `schema` is set to the `XLSFile` schema.
        """
        super().__init__(path=path, dataset_id=dataset_id, schema=XLSFile)
        assert all([filename.endswith(tuple(constants.XLS_EXTENSIONS)) for filename in self.filepaths])

    def get_item(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename, contents, sheet names, and the number of sheets of the XLS file at the
        specified `idx`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary with the filename, contents, sheet names, and the number of sheets of the XLS file.

            .. code-block:: python

                {
                    "filename": "file.xls",
                    "contents": b"raw XLS content here",
                    "sheet_names": ["Sheet1", "Sheet2", "Sheet3],
                    "number_sheets": 3,
                }
        """
        filepath = self.filepaths[idx]
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            contents = f.read()

        xls = pd.ExcelFile(BytesIO(contents), engine="openpyxl")

        return {
            "filename": filename,
            "contents": contents,
            "sheet_names": xls.sheet_names,
            "number_sheets": len(xls.sheet_names),
        }
