import os

import pytest

from palimpzest.core.data.datareaders import DataReader, TextFileDirectoryReader
from palimpzest.core.lib.fields import ImageFilepathField, ListField
from palimpzest.core.lib.schemas import Number

### Raw DataReaders ###
real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": ListField(ImageFilepathField), "desc": "A list of the filepaths for each image of the listing"},
]

class RealEstateListingReader(DataReader):
    def __init__(self, listings_dir):
        super().__init__(real_estate_listing_cols)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)
    
    def __getitem__(self, idx: int):
        # get listing
        listing = self.listings[idx]

        # get fields
        image_filepaths, text_content = [], None
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                image_filepaths.append(os.path.join(listing_dir, file))

        # construct and return dictionary with fields
        return {"listing": listing, "text_content": text_content, "image_filepaths": image_filepaths}


class CostModelTestReader(DataReader):
    def __init__(self):
        super().__init__(Number)
        self.numbers = [1, 2, 3]

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx: int):
        # fetch number
        number = self.numbers[idx]

        # create and return item
        return {"value": number}


### DATA PATH FIXTURES ###
@pytest.fixture
def enron_eval_tiny_data_path():
    return "testdata/enron-eval-tiny"


@pytest.fixture
def real_estate_eval_tiny_data_path():
    return "testdata/real-estate-eval-tiny"


### DATAREADER FIXTURES ###
@pytest.fixture
def enron_eval_tiny(enron_eval_tiny_data_path):
    return TextFileDirectoryReader(enron_eval_tiny_data_path)


@pytest.fixture
def real_estate_eval_tiny(real_estate_eval_tiny_data_path):
    return RealEstateListingReader(real_estate_eval_tiny_data_path)


@pytest.fixture
def cost_model_test_dataset():
    return CostModelTestReader()
