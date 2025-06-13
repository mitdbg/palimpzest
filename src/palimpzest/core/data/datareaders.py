from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO

import pandas as pd
from bs4 import BeautifulSoup

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
class DataReader(ABC):
    """
    The `DataReader` is a base class for which may be used to generate data that
    is processed by PZ.

    Subclasses of the (abstract) `DataReader` class must implement two methods:

    - `__len__()`: which returns the number of elements in the data source
    - `__getitem__(idx: int)`: which takes in an `idx` and returns the element at that index
    """

    def __init__(self, schema: type[Schema] | list[dict]) -> None:
        """
            Constructor for the `DataReader` class.

            Args:
                schema (Schema | list[dict]): The output schema of the records returned by the DataReader
        """
        # NOTE: _schema attribute currently has to match attribute name in Dataset
        self._schema = Schema.from_json(schema) if isinstance(schema, list) else schema

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    @property
    def schema(self) -> Schema:
        return self._schema

    # NOTE: currently used by optimizer to compute node id for DataReaders
    def serialize(self) -> dict:
        return {"schema": self._schema.json_schema()}

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the data reader."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single item from the data reader at the given index.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary representing the item at the given index. The dictionary
                  keys (i.e. fields) should match the fields specified in the schema of the
                  data source, and the values should be the values associated with those fields.

                    # Example return value
                    {"field1": value1, "field2": value2, ...}

        """
        pass


# Second level of abstraction
class DirectoryReader(DataReader):
    """
    DirectoryReader returns a dictionary for each file in a directory. Each dictionary contains the filename and
    contents of a single file in the directory.
    """

    def __init__(self, path: str, schema: Schema) -> None:
        """
        Constructor for the `DirectoryReader` class.

        Args:
            path (str): The path to the directory
            schema (Schema): The output schema of the data source
        """
        assert os.path.isdir(path), f"Path {path} is not a directory"

        self.filepaths = [
            os.path.join(path, filename)
            for filename in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, filename))
        ]
        self.path = path
        super().__init__(schema)

    def serialize(self) -> dict:
        return {
            "schema": self.schema.json_schema(),
            "path": self.path,
            "source_type": "directory",
        }

    def __len__(self) -> int:
        return len(self.filepaths)


class FileReader(DataReader):
    """FileReader returns a single dictionary with the filename and contents of a local file (in bytes)."""

    def __init__(self, path: str) -> None:
        """
        Constructor for the `FileReader` class. The `schema` is set to the default `File` schema.

        Args:
            path (str): The path to the file
        """
        super().__init__(File)
        self.filepath = path

    def serialize(self) -> dict:
        return {
            "schema": self.schema.json_schema(),
            "path": self.filepath,
            "source_type": "file",
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
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


class MemoryReader(DataReader):
    """
    MemoryReader returns one or more dictionaries that reflect the contents of an in-memory Python object `vals`.
    If `vals` is not a pd.DataFrame, then the dictionary returned by `__getitem__()` has a single field called "value".
    Otherwise, the dictionary contains the key-value mapping from columns to values for the `idx` row in the dataframe.

    TODO(gerardo): Add support for other types of in-memory data structures (he has some code for subclassing
        MemoryReader on his branch)
    """

    def __init__(self, vals: list | pd.DataFrame) -> None:
        """
        Constructor for the `MemoryReader` class. The `schema` is set to the default `DefaultSchema` schema.
        If `vals` is a pd.DataFrame, then the schema is set to the schema inferred from the DataFrame.

        Args:
            vals (Any): The in-memory object to use as the data source
        """
        # if list[dict] --> convert to pd.DataFrame first
        self.vals = pd.DataFrame(vals) if isinstance(vals, list) and all([isinstance(item, dict) for item in vals]) else vals
        schema = Schema.from_df(self.vals) if isinstance(self.vals, pd.DataFrame) else DefaultSchema
        super().__init__(schema)

    def __len__(self) -> int:
        return len(self.vals)

    def __getitem__(self, idx: int) -> dict:
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
                {"name": "Alice", "job": "doctor", "hobby": "tennis"}
        """
        item = (
            self.vals.iloc[idx].to_dict()
            if isinstance(self.vals, pd.DataFrame)
            else {"value": self.vals[idx]}
        )

        return item


# Third level of abstraction
class HTMLFileDirectoryReader(DirectoryReader):
    """
    HTMLFileDirectoryReader returns a dictionary for each HTML file in a directory. Each dictionary contains the
    filename, raw HTML content, and parsed content of a single HTML file in the directory.
    """
    def __init__(self, path: str) -> None:
        """
        Constructor for the `HTMLFileDirectoryReader` class. The `schema` is set to the `WebPage` schema.

        Args:
            path (str): The path to the directory
        """
        super().__init__(path=path, schema=WebPage)
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

    def __getitem__(self, idx: int) -> dict:
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


class ImageFileDirectoryReader(DirectoryReader):
    """
    ImageFileDirectoryReader returns a dictionary for each image file in a directory. Each dictionary contains the
    filename and the base64 encoded bytes content of a single image file in the directory.
    """
    def __init__(self, path: str) -> None:
        """
        Constructor for the `ImageFileDirectoryReader` class. The `schema` is set to the `ImageFile` schema.

        Args:
            path (str): The path to the directory
        """
        super().__init__(path=path, schema=ImageFile)
        assert all([filename.endswith(tuple(constants.IMAGE_EXTENSIONS)) for filename in self.filepaths])

    def __getitem__(self, idx: int) -> dict:
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


class PDFFileDirectoryReader(DirectoryReader):
    """
    PDFFileDirectoryReader returns a dictionary for each PDF file in a directory. Each dictionary contains the
    filename, raw PDF content, and parsed text content of a single PDF file in the directory.

    This class also uses one of a predefined set of PDF processors to extract text content from the PDF files.
    """
    def __init__(
        self,
        path: str,
        pdfprocessor: str = "pypdf",
        file_cache_dir: str = "/tmp",
    ) -> None:
        """
        Constructor for the `PDFFileDirectoryReader` class. The `schema` is set to the `PDFFile` schema.

        Args:
            path (str): The path to the directory
            pdfprocessor (str): The PDF processor to use for extracting text content from the PDF files
            file_cache_dir (str): The directory to store the temporary files generated during PDF processing
        """
        super().__init__(path=path, schema=PDFFile)
        assert all([filename.endswith(tuple(constants.PDF_EXTENSIONS)) for filename in self.filepaths])
        self.pdfprocessor = pdfprocessor
        self.file_cache_dir = file_cache_dir

    def __getitem__(self, idx: int) -> dict:
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

        # generate text_content from PDF
        text_content = get_text_from_pdf(pdf_filename, pdf_bytes, pdfprocessor=self.pdfprocessor, file_cache_dir=self.file_cache_dir)

        # construct and return item
        return {"filename": pdf_filename, "contents": pdf_bytes, "text_contents": text_content}


class TextFileDirectoryReader(DirectoryReader):
    """
    TextFileDirectoryReader returns a dictionary for each text file in a directory. Each dictionary contains the
    filename and contents of a single text file in the directory.
    """
    def __init__(self, path: str) -> None:
        """
        Constructor for the `TextFileDirectoryReader` class. The `schema` is set to the `TextFile` schema.

        Args:
            path (str): The path to the directory
        """
        super().__init__(path=path, schema=TextFile)

    def __getitem__(self, idx: int) -> dict:
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


class XLSFileDirectoryReader(DirectoryReader):
    """
    XLSFileDirectoryReader returns a dictionary for each XLS file in a directory. Each dictionary contains the
    filename, contents, sheet names, and the number of sheets for a single XLS file in the directory.
    """
    def __init__(self, path: str) -> None:
        """
        Constructor for the `XLSFileDirectoryReader` class. The `schema` is set to the `XLSFile` schema.
        """
        super().__init__(path=path, schema=XLSFile)
        assert all([filename.endswith(tuple(constants.XLS_EXTENSIONS)) for filename in self.filepaths])

    def __getitem__(self, idx: int) -> dict:
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
