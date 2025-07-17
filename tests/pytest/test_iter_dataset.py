import os
from copy import deepcopy

import pandas as pd
import pytest

from palimpzest.core.data.iter_dataset import (
    HTMLFileDataset,
    ImageFileDataset,
    MemoryDataset,
    TextFileDataset,
)
from palimpzest.core.lib.schemas import TextFile, WebPage


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


def test_text_dataset(temp_text_dir):
    dataset = TextFileDataset(id="test", path=temp_text_dir)
    assert len(dataset) == 2
    assert dataset.schema == TextFile
    
    record = dataset[0]
    assert isinstance(record, dict)
    assert record["contents"] == "Content 1"
    
    record = dataset[1]
    assert record["contents"] == "Content 2"

def test_memory_dataset_list(list_values):
    dataset = MemoryDataset(id="test", vals=list_values)
    assert len(dataset) == len(list_values)
    
    record = dataset[0]
    assert record["value"] == list_values[0]
    record = dataset[3]
    assert record["value"] == list_values[3]
    copied = deepcopy(dataset)
    assert copied.vals == dataset.vals

def test_memory_dataset_df(df_values):
    dataset = MemoryDataset(id="test", vals=df_values)
    assert len(dataset) == len(df_values)
    
    record = dataset[0]
    assert record["a"] == df_values.iloc[0]['a']
    assert record["b"] == df_values.iloc[0]['b']

    copied = deepcopy(dataset)
    assert copied.vals.equals(dataset.vals)


def test_memory_dataset_copy():
    values = [1, 2, 3]
    dataset = MemoryDataset(id="test", vals=values)
    copied = deepcopy(dataset)
    
    assert copied.vals == dataset.vals

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

def test_html_dataset(temp_html_dir):
    dataset = HTMLFileDataset(id="test", path=temp_html_dir)
    assert len(dataset) == 1
    assert dataset.schema == WebPage
    
    record = dataset[0]
    assert isinstance(record, dict)
    assert "Example Link (http://example.com)" in record["text"]
    assert "<html>" in record["html"]

def test_invalid_directory():
    with pytest.raises(AssertionError):
        ImageFileDataset(id="test", path="/nonexistent/path")
