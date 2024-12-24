import os
from pathlib import Path

import pytest

import palimpzest as pz


### DATA SOURCES ###
# NOTE: I need to have RealEstateListingFiles and RealEstateListingSource
#       outside of a fixture here in order for the DataDirectory to properly
#       pickle user datasources.
class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""

    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.StringField(desc="The content of the listing's text description", required=True)
    image_filepaths = pz.ListField(
        element_type=pz.StringField,
        desc="A list of the filepaths for each image of the listing",
        required=True,
    )


class RealEstateListingSource(pz.UserSource):
    def __init__(self, dataset_id, listings_dir):
        super().__init__(RealEstateListingFiles, dataset_id)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def copy(self):
        return RealEstateListingSource(self.dataset_id, self.listings_dir)

    def __len__(self):
        return len(self.listings)

    def get_size(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob("*"))

    def get_item(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = pz.DataRecord(self.schema, source_id=listing)
        dr.listing = listing
        dr.image_filepaths = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            if file.endswith(".txt"):
                with open(os.path.join(listing_dir, file), "rb") as f:
                    dr.text_content = f.read().decode("utf-8")
            elif file.endswith(".png"):
                dr.image_filepaths.append(os.path.join(listing_dir, file))

        return dr

class CostModelTestSource(pz.UserSource):
    def __init__(self, datasetId):
        super().__init__(pz.Number, datasetId)
        self.numbers = [1, 2, 3]

    def copy(self):
        return CostModelTestSource(self.dataset_id)

    def __len__(self):
        return len(self.numbers)

    def get_size(self):
        return 0

    def get_item(self, idx: int):
        # fetch number
        number = self.numbers[idx]

        # create data record
        dr = pz.DataRecord(self.schema, source_id=idx)
        dr.value = number

        return dr


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
    pz.DataDirectory().register_local_directory(
        path=enron_eval_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id


@pytest.fixture
def real_estate_eval_tiny(real_estate_eval_tiny_data):
    dataset_id = "real-estate-eval-tiny"

    pz.DataDirectory().register_user_source(
        src=RealEstateListingSource(dataset_id, real_estate_eval_tiny_data),
        dataset_id=dataset_id,
    )
    yield dataset_id


@pytest.fixture
def biofabric_tiny(biofabric_tiny_data):
    dataset_id = "biofabric-tiny"
    pz.DataDirectory().register_local_directory(
        path=biofabric_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id 


@pytest.fixture
def cost_model_test_dataset():
    dataset_id = "cost-model-test-dataset"

    pz.DataDirectory().register_user_source(
        src=CostModelTestSource(dataset_id),
        dataset_id=dataset_id,
    )
    yield dataset_id
