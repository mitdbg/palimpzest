import pytest
import palimpzest as pz
import os
from pathlib import Path

### DATA SOURCES ###
# NOTE: I need to have RealEstateListingFiles and RealEstateListingSource
#       outside of a fixture here in order for the DataDirectory to properly
#       pickle user datasources.
class RealEstateListingFiles(pz.Schema):
    """The source text and image data for a real estate listing."""

    listing = pz.StringField(desc="The name of the listing", required=True)
    text_content = pz.StringField(
        desc="The content of the listing's text description", required=True
    )
    image_contents = pz.ListField(
        element_type=pz.BytesField,
        desc="A list of the contents of each image of the listing",
        required=True,
    )

class RealEstateListingSource(pz.UserSource):
    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.listings = sorted(os.listdir(self.listings_dir))

    def __len__(self):
        return len(self.listings)

    def getSize(self):
        return sum(file.stat().st_size for file in Path(self.listings_dir).rglob('*'))

    def getItem(self, idx: int):
        # fetch listing
        listing = self.listings[idx]

        # create data record
        dr = pz.DataRecord(self.schema, scan_idx=idx)
        dr.listing = listing
        dr.image_contents = []
        listing_dir = os.path.join(self.listings_dir, listing)
        for file in os.listdir(listing_dir):
            bytes_data = None
            with open(os.path.join(listing_dir, file), "rb") as f:
                bytes_data = f.read()
            if file.endswith(".txt"):
                dr.text_content = bytes_data.decode("utf-8")
            elif file.endswith(".png"):
                dr.image_contents.append(bytes_data)

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
    pz.DataDirectory().registerLocalDirectory(
        path=enron_eval_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id

@pytest.fixture
def real_estate_eval_tiny(real_estate_eval_tiny_data):
    dataset_id = "real-estate-eval-tiny"

    pz.DataDirectory().registerUserSource(
        src=RealEstateListingSource(dataset_id, real_estate_eval_tiny_data),
        dataset_id=dataset_id,
    )
    yield dataset_id

@pytest.fixture
def biofabric_tiny(biofabric_tiny_data):
    dataset_id = "biofabric-tiny"
    pz.DataDirectory().registerLocalDirectory(
        path=biofabric_tiny_data,
        dataset_id=dataset_id,
    )
    yield dataset_id 
