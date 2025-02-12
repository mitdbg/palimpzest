import os
from copy import deepcopy

import pandas as pd
import pytest

from palimpzest.core.data.datareaders import (
    FileReader,
    HTMLFileDirectoryReader,
    ImageFileDirectoryReader,
    MemoryReader,
    TextFileDirectoryReader,
)
from palimpzest.core.lib.schemas import File, TextFile, WebPage


@pytest.fixture
def temp_text_file():
    file_path = "testdata/tmp_test.txt"
    with open(file_path, "w") as f:
        f.write("Hello, World!")
    yield file_path
    os.remove(file_path)

@pytest.fixture
def temp_text_dir():
    dir_path = "testdata/text_dir"
    os.makedirs(dir_path, exist_ok=True)
    with open(dir_path + "/file1.txt", "w") as f:
        f.write("Content 1")
    with open(dir_path + "/file2.txt", "w") as f:
        f.write("Content 2")
    yield dir_path
    os.remove("testdata/text_dir/file1.txt")
    os.remove("testdata/text_dir/file2.txt")
    os.rmdir(dir_path)

@pytest.fixture
def list_values():
    return [1, 2, 3, 4]

@pytest.fixture
def df_values():
    return pd.DataFrame({"a": [10, 20, 30, 40], "b": [50, 60, 70, 80]})


def test_file_reader_initialization(temp_text_file):
    reader = FileReader(temp_text_file)
    assert reader.filepath == temp_text_file
    assert reader.schema == File

def test_file_reader(temp_text_file):
    reader = FileReader(temp_text_file)
    record = reader[0]
    
    assert isinstance(record, dict)
    assert record["filename"] == temp_text_file
    assert record["contents"] == b"Hello, World!"
    assert len(reader) == 1

    copied = deepcopy(reader)
    assert copied.filepath == reader.filepath
    assert copied.schema == reader.schema

def test_text_directory_reader(temp_text_dir):
    reader = TextFileDirectoryReader(temp_text_dir)
    assert len(reader) == 2
    assert reader.schema == TextFile
    
    record = reader[0]
    assert isinstance(record, dict)
    assert record["contents"] == "Content 1"
    
    record = reader[1]
    assert record["contents"] == "Content 2"

def test_memory_reader_list(list_values):
    reader = MemoryReader(list_values)
    assert len(reader) == len(list_values)
    
    record = reader[0]
    assert record["value"] == list_values[0]
    record = reader[3]
    assert record["value"] == list_values[3]
    copied = deepcopy(reader)
    assert copied.vals == reader.vals

def test_memory_reader_df(df_values):
    reader = MemoryReader(df_values)
    assert len(reader) == len(df_values)
    
    record = reader[0]
    assert record["a"] == df_values.iloc[0]['a']
    assert record["b"] == df_values.iloc[0]['b']

    copied = deepcopy(reader)
    assert copied.vals.equals(reader.vals)


def test_memory_reader_copy():
    values = [1, 2, 3]
    reader = MemoryReader(values)
    copied = deepcopy(reader)
    
    assert copied.vals == reader.vals

@pytest.fixture
def temp_html_dir(tmp_path):
    dir_path = tmp_path / "html_files"
    dir_path.mkdir()
    html_content = """
    <html>
        <body>
            <a href="http://example.com">Example Link</a>
            <p>Some text</p>
        </body>
    </html>
    """
    (dir_path / "page1.html").write_text(html_content)
    return str(dir_path)

def test_html_directory_reader(temp_html_dir):
    reader = HTMLFileDirectoryReader(temp_html_dir)
    assert len(reader) == 1
    assert reader.schema == WebPage
    
    record = reader[0]
    assert isinstance(record, dict)
    assert "Example Link (http://example.com)" in record["text"]
    assert "<html>" in record["html"]

def test_invalid_directory():
    with pytest.raises(AssertionError):
        ImageFileDirectoryReader("/nonexistent/path")

def test_reader_serialization(temp_text_file):
    reader = FileReader(temp_text_file)
    serialized = reader.serialize()
    
    assert "schema" in serialized
    assert "path" in serialized
    assert "source_type" in serialized
    assert serialized["source_type"] == "file"
