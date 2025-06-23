from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from palimpzest import constants
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import (
    DefaultSchema,
    ImageFile,
    PDFFile,
    Schema,
    TextFile,
    WebPage,
    XLSFile,
)
from palimpzest.query.operators.logical import BaseScan
from palimpzest.tools.pdfparser import get_text_from_pdf


####################
### BASE CLASSES ###
####################
class IterDataset(Dataset, ABC):
    """
    The `IterDataset` is an abstract base class for root `Datasets` whose data is accessed
    via iteration. Classes which inherit from this class must implement two methods:

    - `__len__()`: which returns the number of elements in the dataset
    - `__getitem__(idx: int)`: which takes in an `idx` and returns the element at that index
    """

    def __init__(self, id: str, schema: type[Schema] | list[dict]) -> None:
        """
            Constructor for the `IterDataset` class.

            Args:
                id (str): a string identifier for the `Dataset`
                schema (Schema | list[dict]): The output schema of the records returned by the `Dataset`
        """
        # set the id for the Dataset
        self._id = id

        # compute Schema and call parent constructor
        schema = Schema.from_json(schema) if isinstance(schema, list) else schema
        super().__init__(sources=None, operator=BaseScan(datasource=self, output_schema=schema), schema=schema)

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of items in the `Dataset`."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single item from the `Dataset` at the given index.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary representing the item at the given index. The dictionary
                  keys (i.e. fields) should match the fields specified in the schema of the
                  dataset, and the values should be the values associated with those fields.

                    # Example return value
                    {"field1": value1, "field2": value2, ...}

        """
        pass


class BaseFileDataset(IterDataset):
    """
    BaseFileDataset is the base class for multiple `IterDatasets` which iterate over
    different types of files.
    """

    def __init__(self, path: str, **kwargs) -> None:
        """
        Constructor for the `BaseFileDataset` class.

        Args:
            path (str): The path to the file
            kwargs (dict): Keyword arguments containing the `Dataset's` id and file-specific `Schema`
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

    def __len__(self) -> int:
        return len(self.filepaths)


########################
### CONCRETE CLASSES ###
########################
class MemoryDataset(IterDataset):
    """
    MemoryDataset returns one or more dictionaries that reflect the contents of an in-memory Python object `vals`.
    If `vals` is not a pd.DataFrame, then the dictionary returned by `__getitem__()` has a single field called "value".
    Otherwise, the dictionary contains the key-value mapping from columns to values for the `idx` row in the dataframe.

    TODO(gerardo): Add support for other types of in-memory data structures (he has some code for subclassing
        MemoryDataset on his branch)
    """

    def __init__(self, id: str, vals: list | pd.DataFrame) -> None:
        """
        Constructor for the `MemoryDataset` class. The `schema` is set to the default `DefaultSchema` schema.
        If `vals` is a pd.DataFrame, then the schema is set to the schema inferred from the DataFrame.

        Args:
            id (str): a string identifier for the `Dataset`
            vals (Any): The in-memory data to iterate over
        """
        # if list[dict] --> convert to pd.DataFrame first
        self.vals = pd.DataFrame(vals) if isinstance(vals, list) and all([isinstance(item, dict) for item in vals]) else vals
        schema = Schema.from_df(self.vals) if isinstance(self.vals, pd.DataFrame) else DefaultSchema
        super().__init__(id=id, schema=schema)

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


class HTMLFileDataset(BaseFileDataset):
    """
    HTMLFileDataset returns a dictionary for each HTML file in a directory. Each dictionary contains the
    filename, raw HTML content, and parsed content of a single HTML file in the directory.
    """
    def __init__(self, id: str, path: str) -> None:
        """
        Constructor for the `HTMLFileDataset` class. The `schema` is set to the `WebPage` schema.

        Args:
            id (str): a string identifier for the `Dataset`
            path (str): The path to the directory
        """
        super().__init__(path=path, id=id, schema=WebPage)
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


class ImageFileDataset(BaseFileDataset):
    """
    ImageFileDataset returns a dictionary for each image file in a directory. Each dictionary contains the
    filename and the base64 encoded bytes content of a single image file in the directory.
    """
    def __init__(self, id: str, path: str) -> None:
        """
        Constructor for the `ImageFileDataset` class. The `schema` is set to the `ImageFile` schema.

        Args:
            id (str): a string identifier for the `Dataset`
            path (str): The path to the directory
        """
        super().__init__(path=path, id=id, schema=ImageFile)
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


class PDFFileDataset(BaseFileDataset):
    """
    PDFFileDataset returns a dictionary for each PDF file in a directory. Each dictionary contains the
    filename, raw PDF content, and parsed text content of a single PDF file in the directory.

    This class also uses one of a predefined set of PDF processors to extract text content from the PDF files.
    """
    def __init__(
        self,
        id: str,
        path: str,
        pdfprocessor: str = "pypdf",
        file_cache_dir: str = "/tmp",
    ) -> None:
        """
        Constructor for the `PDFFileDataset` class. The `schema` is set to the `PDFFile` schema.

        Args:
            id (str): a string identifier for the `Dataset`
            path (str): The path to the directory
            pdfprocessor (str): The PDF processor to use for extracting text content from the PDF files
            file_cache_dir (str): The directory to store the temporary files generated during PDF processing
        """
        super().__init__(path=path, id=id, schema=PDFFile)
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


class TextFileDataset(BaseFileDataset):
    """
    TextFileDataset returns a dictionary for each text file in a directory. Each dictionary contains the
    filename and contents of a single text file in the directory.
    """
    def __init__(self, id: str, path: str) -> None:
        """
        Constructor for the `TextFileDataset` class. The `schema` is set to the `TextFile` schema.

        Args:
            id (str): a string identifier for the `Dataset`
            path (str): The path to the directory
        """
        super().__init__(path=path, id=id, schema=TextFile)

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


class XLSFileDataset(BaseFileDataset):
    """
    XLSFileDataset returns a dictionary for each XLS file in a directory. Each dictionary contains the
    filename, contents, sheet names, and the number of sheets for a single XLS file in the directory.
    """
    def __init__(self, id: str, path: str) -> None:
        """
        Constructor for the `XLSFileDataset` class. The `schema` is set to the `XLSFile` schema.
        """
        super().__init__(path=path, id=id, schema=XLSFile)
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


def get_local_source(path: str | Path, **kwargs) -> Dataset:
    """Return a `Dataset` for a local file or directory."""
    if os.path.isfile(path):
        return TextFileDataset(path)

    elif os.path.isdir(path):
        if all([f.endswith(tuple(constants.IMAGE_EXTENSIONS)) for f in os.listdir(path)]):
            return ImageFileDataset(path)

        elif all([f.endswith(tuple(constants.PDF_EXTENSIONS)) for f in os.listdir(path)]):
            pdfprocessor = kwargs.get("pdfprocessor", constants.DEFAULT_PDF_PROCESSOR)
            file_cache_dir = kwargs.get("file_cache_dir", "/tmp")
            return PDFFileDataset(
                path=path, pdfprocessor=pdfprocessor, file_cache_dir=file_cache_dir
            )

        elif all([f.endswith(tuple(constants.XLS_EXTENSIONS)) for f in os.listdir(path)]):
            return XLSFileDataset(path)

        elif all([f.endswith(tuple(constants.HTML_EXTENSIONS)) for f in os.listdir(path)]):
            return HTMLFileDataset(path)

        else:
            return TextFileDataset(path)
    else:
        raise ValueError(f"Path {path} is invalid. Does not point to a file or directory.")


def resolve_datasource(source: str | Path | list | pd.DataFrame, **kwargs) -> Dataset:
    """
    This helper function returns a `Dataset` object based on the `source` type.
    The returned `Dataset` object is guaranteed to have a schema.
    """
    if isinstance(source, (str, Path)):
        source = get_local_source(source, **kwargs)

    elif isinstance(source, (list, pd.DataFrame)):
        source = MemoryDataset(source)

    else:
        raise ValueError(f"Invalid source type: {type(source)}, We only support str, Path, list[dict], and pd.DataFrame")

    return source
