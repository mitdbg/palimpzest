import os
from pathlib import Path

import pandas as pd

from palimpzest import constants
from palimpzest.core.data.datareaders import (
    DataReader,
    FileReader,
    HTMLFileDirectoryReader,
    ImageFileDirectoryReader,
    MemoryReader,
    PDFFileDirectoryReader,
    TextFileDirectoryReader,
    XLSFileDirectoryReader,
)


def get_local_source(path: str | Path, **kwargs) -> DataReader:
    """Return a DataReader for a local file or directory."""
    if os.path.isfile(path):
        return FileReader(path)

    elif os.path.isdir(path):
        if all([f.endswith(tuple(constants.IMAGE_EXTENSIONS)) for f in os.listdir(path)]):
            return ImageFileDirectoryReader(path)

        elif all([f.endswith(tuple(constants.PDF_EXTENSIONS)) for f in os.listdir(path)]):
            pdfprocessor = kwargs.get("pdfprocessor", constants.DEFAULT_PDF_PROCESSOR)
            file_cache_dir = kwargs.get("file_cache_dir", "/tmp")
            return PDFFileDirectoryReader(
                path=path, pdfprocessor=pdfprocessor, file_cache_dir=file_cache_dir
            )

        elif all([f.endswith(tuple(constants.XLS_EXTENSIONS)) for f in os.listdir(path)]):
            return XLSFileDirectoryReader(path)

        elif all([f.endswith(tuple(constants.HTML_EXTENSIONS)) for f in os.listdir(path)]):
            return HTMLFileDirectoryReader(path)

        else:
            return TextFileDirectoryReader(path)
    else:
        raise Exception(f"Path {path} is invalid. Does not point to a file or directory.")


def get_local_datareader(source: str | Path | list | pd.DataFrame, **kwargs) -> DataReader:
    """
    This helper function returns a `DataReader` object based on the `source` type.
    The returned `DataReader` object is guaranteed to have a schema.
    """
    if isinstance(source, (str, Path)):
        source = get_local_source(source, **kwargs)

    elif isinstance(source, (list, pd.DataFrame)):
        source = MemoryReader(source)

    else:
        raise Exception(f"Invalid source type: {type(source)}, We only support str, Path, list[dict], and pd.DataFrame")

    return source
