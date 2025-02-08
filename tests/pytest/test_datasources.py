# write tests for src/palimpzest/core/data/datasources.py

import os
from copy import deepcopy

import pandas as pd
import pytest

from palimpzest.core.data.datareaders import (
    FileSource,
    HTMLFileDirectorySource,
    ImageFileDirectorySource,
    MemorySource,
    TextFileDirectorySource,
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


def test_file_source_initialization(temp_text_file):
    source = FileSource(temp_text_file, "test_dataset")
    assert source.filepath == temp_text_file
    assert source.dataset_id == "test_dataset"
    assert source.schema == File

def test_file_source(temp_text_file):
    source = FileSource(temp_text_file, "test_dataset")
    record = source.get_item(0)
    
    assert isinstance(record, dict)
    assert record["filename"] == temp_text_file
    assert record["contents"] == b"Hello, World!"
    assert len(source) == 1

    copied = deepcopy(source)
    assert copied.filepath == source.filepath
    assert copied.dataset_id == source.dataset_id
    assert copied.schema == source.schema

def test_text_directory_source(temp_text_dir):
    source = TextFileDirectorySource(temp_text_dir, "test_dataset")
    assert len(source) == 2
    assert source.schema == TextFile
    
    record = source.get_item(0)
    assert isinstance(record, dict)
    assert record["contents"] == "Content 1"
    
    record = source.get_item(1)
    assert record["contents"] == "Content 2"

def test_memory_source_list(list_values):
    source = MemorySource(list_values, dataset_id="test_memory")
    assert len(source) == len(list_values)
    assert source.dataset_id == "test_memory"
    
    record = source.get_item(0)
    assert record["value"] == list_values[0]
    record = source.get_item(3)
    assert record["value"] == list_values[3]
    copied = deepcopy(source)
    assert copied.vals == source.vals
    assert copied.dataset_id == source.dataset_id

def test_memory_source_df(df_values):
    source = MemorySource(df_values, dataset_id="test_memory")
    assert len(source) == len(df_values)
    assert source.dataset_id == "test_memory"
    
    record = source.get_item(0)
    assert record["a"] == df_values.iloc[0]['a']
    assert record["b"] == df_values.iloc[0]['b']

    copied = deepcopy(source)
    assert copied.vals.equals(source.vals)
    assert copied.dataset_id == source.dataset_id


def test_memory_source_copy():
    values = [1, 2, 3]
    source = MemorySource(values, dataset_id="test_memory")
    copied = deepcopy(source)
    
    assert copied.vals == source.vals
    assert copied.dataset_id == source.dataset_id

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

def test_html_directory_source(temp_html_dir):
    source = HTMLFileDirectorySource(temp_html_dir, "test_dataset")
    assert len(source) == 1
    assert source.schema == WebPage
    
    record = source.get_item(0)
    assert isinstance(record, dict)
    assert "Example Link (http://example.com)" in record["text"]
    assert "<html>" in record["html"]

def test_invalid_directory():
    with pytest.raises(AssertionError):
        ImageFileDirectorySource("/nonexistent/path", "test_dataset")

def test_source_serialization(temp_text_file):
    source = FileSource(temp_text_file, "test_dataset")
    serialized = source.serialize()
    
    assert "schema" in serialized
    assert "path" in serialized
    assert "source_type" in serialized
    assert serialized["source_type"] == "file"