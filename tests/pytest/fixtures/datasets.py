import os

import pytest

from palimpzest.core.data.datareaders import DataSource
from palimpzest.core.lib.fields import ImageFilepathField, ListField, NumericField
from palimpzest.core.lib.schemas import Schema
from palimpzest.datamanager.datamanager import DataDirectory

### DATA SOURCES ###
# NOTE: I need to have RealEstateListingFiles and RealEstateListingSource
#       outside of a fixture here in order for the DataDirectory to properly
#       pickle user datasources.
real_estate_listing_cols = [
    {"name": "listing", "type": str, "desc": "The name of the listing"},
    {"name": "text_content", "type": str, "desc": "The content of the listing's text description"},
    {"name": "image_filepaths", "type": ListField(ImageFilepathField), "desc": "A list of the filepaths for each image of the listing"},
]
# class RealEstateListingFiles(Schema):
#     """The source text and image data for a real estate listing."""

#     listing = StringField(desc="The name of the listing")
#     text_content = StringField(desc="The content of the listing's text description")
#     image_filepaths = ListField(
#         element_type=StringField,
#         desc="A list of the filepaths for each image of the listing",
#     )

class Number(Schema):
    value = NumericField(desc="The value of the number")


class RealEstateListingSource(DataSource):
    def __init__(self, dataset_id, listings_dir):
        super().__init__(real_estate_listing_cols, dataset_id)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)
    
    def get_item(self, idx: int):
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

class CostModelTestSource(DataSource):
    def __init__(self, dataset_id: str):
        super().__init__(Number, dataset_id)
        self.numbers = [1, 2, 3]

    def __len__(self):
        return len(self.numbers)

    def get_item(self, idx: int):
        # fetch number
        number = self.numbers[idx]

        # create and return item
        return {"value": number}


### DATASET DATA PATHS ###
@pytest.fixture
def enron_eval_tiny_data():
    return "testdata/enron-eval-tiny"


@pytest.fixture
def real_estate_eval_tiny_data():
    return "testdata/real-estate-eval-tiny"


@pytest.fixture
def biofabric_tiny_data():
    return "testdata/biofabric-tiny"


### DATASETS ###
@pytest.fixture
def enron_eval_tiny(enron_eval_tiny_data):
    dataset_id = "enron-eval-tiny"
    DataDirectory().register_local_directory(
        path=enron_eval_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id


@pytest.fixture
def real_estate_eval_tiny(real_estate_eval_tiny_data):
    dataset_id = "real-estate-eval-tiny"

    DataDirectory().register_user_source(
        src=RealEstateListingSource(dataset_id, real_estate_eval_tiny_data),
        dataset_id=dataset_id,
    )
    yield dataset_id


@pytest.fixture
def biofabric_tiny(biofabric_tiny_data):
    dataset_id = "biofabric-tiny"
    DataDirectory().register_local_directory(
        path=biofabric_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id 


@pytest.fixture
def cost_model_test_dataset():
    dataset_id = "cost-model-test-dataset"

    DataDirectory().register_user_source(
        src=CostModelTestSource(dataset_id),
        dataset_id=dataset_id,
    )
    yield dataset_id
