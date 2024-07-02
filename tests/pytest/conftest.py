from pathlib import Path
import palimpzest as pz
import os
import pytest

# TODO: in the future I will register these datasets as part of test class setup
# DEFINITIONS
ENRON_EVAL_TINY_TEST_DATA = "testdata/enron-eval-tiny"
ENRON_EVAL_TINY_DATASET_ID = "enron-eval-tiny"
REAL_ESTATE_EVAL_TINY_TEST_DATA = "testdata/real-estate-eval-tiny"
REAL_ESTATE_EVAL_TINY_DATASET_ID = "real-estate-eval-tiny"
BIOFABRIC_EVAL_TINY_TEST_DATA = "testdata/biofabric-tiny"
BIOFABRIC_EVAL_TINY_DATASET_ID = "biofabric-tiny"

# Addresses far from MIT; we use a simple lookup like this to make the
# experiments re-producible w/out needed a Google API key for geocoding lookups
FAR_AWAY_ADDRS = [
    "Melcher St",
    "Sleeper St",
    "437 D St",
    "Seaport Blvd",
    "50 Liberty Dr",
    "Telegraph St",
    "Columbia Rd",
    "E 6th St",
    "E 7th St",
    "E 5th St",
]


with open(".env") as f:
    for line in f:
        key, value = line.strip().split("=")
        os.environ[key] = value

@pytest.fixture
def email_schema():
    class Email(pz.TextFile):
        """Represents an email, which in practice is usually from a text file"""

        sender = pz.Field(desc="The email address of the sender", required=True)
        subject = pz.Field(desc="The subject of the email", required=True)
    
    return Email


@pytest.fixture
def real_estate_listing_files_schema():
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

    return RealEstateListingFiles

@pytest.fixture
def image_real_estate_listing_schema(real_estate_listing_files_schema):
    class ImageRealEstateListing(real_estate_listing_files_schema):
        """Represents a real estate listing with specific fields extracted from its text and images."""

        is_modern_and_attractive = pz.BooleanField(
            desc="True if the home interior design is modern and attractive and False otherwise"
        )
        has_natural_sunlight = pz.BooleanField(
            desc="True if the home interior has lots of natural sunlight and False otherwise"
        )

    return ImageRealEstateListing


@pytest.fixture
def real_estate_listing_datasource(real_estate_listing_files_schema):
    class RealEstateListingSource(pz.UserSource):
        def __init__(self, datasetId, listings_dir):
            super().__init__(real_estate_listing_files_schema, datasetId)
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

    datasetIdentifier = REAL_ESTATE_EVAL_TINY_DATASET_ID
    datadir = pz.DataDirectory()
    datadir.registerUserSource(
        RealEstateListingSource(datasetIdentifier, REAL_ESTATE_EVAL_TINY_TEST_DATA), datasetIdentifier
    )

    return RealEstateListingSource


@pytest.fixture
def case_data_schema():
    class CaseData(pz.Schema):
        """An individual row extracted from a table containing medical study data."""

        case_submitter_id = pz.Field(desc="The ID of the case", required=True)
        age_at_diagnosis = pz.Field(
            desc="The age of the patient at the time of diagnosis", required=False
        )
        race = pz.Field(
            desc="An arbitrary classification of a taxonomic group that is a division of a species.",
            required=False,
        )
        ethnicity = pz.Field(
            desc="Whether an individual describes themselves as Hispanic or Latino or not.",
            required=False,
        )
        gender = pz.Field(desc="Text designations that identify gender.", required=False)
        vital_status = pz.Field(desc="The vital status of the patient", required=False)
        ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
        ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
        ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
        tumor_grade = pz.Field(desc="The tumor grade", required=False)
        tumor_focality = pz.Field(desc="The tumor focality", required=False)
        tumor_largest_dimension_diameter = pz.Field(
            desc="The tumor largest dimension diameter", required=False
        )
        primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
        morphology = pz.Field(desc="The morphology", required=False)
        tissue_or_organ_of_origin = pz.Field(
            desc="The tissue or organ of origin", required=False
        )
        # tumor_code = pz.Field(desc="The tumor code", required=False)
        filename = pz.Field(
            desc="The name of the file the record was extracted from", required=False
        )
        study = pz.Field(
            desc="The last name of the author of the study, from the table name",
            required=False,
        )

    return CaseData


@pytest.fixture(scope="class")
def emails_dataset():
    datasetIdentifier = ENRON_EVAL_TINY_DATASET_ID
    datadir = pz.DataDirectory()
    datadir.registerLocalDirectory(ENRON_EVAL_TINY_TEST_DATA, datasetIdentifier)

    return datasetIdentifier


@pytest.fixture
def enron_eval(email_schema):
    emails = pz.Dataset(ENRON_EVAL_TINY_DATASET_ID, schema=email_schema)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails

### Real Estate Listing Workloads ###
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

class TextRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text."""

    address = pz.StringField(desc="The address of the property")
    price = pz.NumericField(desc="The listed price of the property")


class ImageRealEstateListing(RealEstateListingFiles):
    """Represents a real estate listing with specific fields extracted from its text and images."""

    is_modern_and_attractive = pz.BooleanField(
        desc="True if the home interior design is modern and attractive and False otherwise"
    )
    has_natural_sunlight = pz.BooleanField(
        desc="True if the home interior has lots of natural sunlight and False otherwise"
    )


# @pytest.fixture
# def real_estate_listing_source():
class RealEstateListingSource(pz.UserSource):

    def __init__(self, datasetId, listings_dir):
        super().__init__(RealEstateListingFiles, datasetId)
        self.listings_dir = listings_dir
        self.idx = 0
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

# return RealEstateListingSource

def within_two_miles_of_mit(record):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    try:
        if any(
            [
                street.lower() in record.address.lower()
                for street in FAR_AWAY_ADDRS
            ]
        ):
            return False
        return True
    except:
        return False

def in_price_range(record):
    try:
        price = record.price
        if type(price) == str:
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price and price <= 2e6
    except:
        return False

@pytest.fixture
def real_estate_eval():
    pz.DataDirectory().registerUserSource(
    RealEstateListingSource('real-estate-eval-tiny', 'testdata/real-estate-eval-tiny'), 'real-estate-eval-tiny')

    listings = pz.Dataset("real-estate-eval-tiny", schema=RealEstateListingFiles)
    listings = listings.convert(TextRealEstateListing, depends_on="text_content")
    listings = listings.convert(ImageRealEstateListing, image_conversion=True, depends_on="image_contents")
    listings = listings.filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    listings = listings.filter(within_two_miles_of_mit, depends_on="address")
    listings = listings.filter(in_price_range, depends_on="price")
    return listings

### Biofabric Workloads ###

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(
        desc="The age of the patient at the time of diagnosis", required=False
    )
    race = pz.Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species.",
        required=False,
    )
    ethnicity = pz.Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not.",
        required=False,
    )
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(
        desc="The tumor largest dimension diameter", required=False
    )
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(
        desc="The tissue or organ of origin", required=False
    )
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    filename = pz.Field(
        desc="The name of the file the record was extracted from", required=False
    )
    study = pz.Field(
        desc="The last name of the author of the study, from the table name",
        required=False,
    )

@pytest.fixture
def biofabric_eval():
    xls = pz.Dataset("biofabric-medium", schema=pz.XLSFile)
    patient_tables = xls.convert(
        pz.Table, desc="All tables in the file", cardinality="oneToMany")
    patient_tables = patient_tables.filter(
        "The rows of the table contain the patient age"
    )
    case_data = patient_tables.convert(
        CaseData, desc="The patient data in the table", cardinality="oneToMany"
    )
    return case_data
