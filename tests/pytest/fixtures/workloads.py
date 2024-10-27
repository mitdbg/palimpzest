import pytest

from palimpzest.constants import Cardinality
from palimpzest.corelib.schemas import Table, XLSFile
from palimpzest.sets import Dataset
from palimpzest.utils import udfs


### UDFs ###
def within_two_miles_of_mit(record):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
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
    try:
        if any([street.lower() in record.address.lower() for street in FAR_AWAY_ADDRS]):
            return False
        return True
    except Exception:
        return False


def in_price_range(record):
    try:
        price = record.price
        if type(price) == str:
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price and price <= 2e6
    except Exception:
        return False


### WORKLOADS ###
@pytest.fixture
def enron_workload(enron_eval_tiny, email_schema):
    emails = Dataset(enron_eval_tiny, schema=email_schema)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails


@pytest.fixture
def real_estate_workload(
    real_estate_eval_tiny,
    real_estate_listing_files_schema,
    text_real_estate_listing_schema,
    image_real_estate_listing_schema,
):
    listings = Dataset(real_estate_eval_tiny, schema=real_estate_listing_files_schema)
    listings = listings.convert(text_real_estate_listing_schema, depends_on="text_content")
    listings = listings.convert(image_real_estate_listing_schema, image_conversion=True, depends_on="image_contents")
    listings = listings.filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    listings = listings.filter(within_two_miles_of_mit, depends_on="address")
    listings = listings.filter(in_price_range, depends_on="price")
    return listings


@pytest.fixture
def biofabric_workload(biofabric_tiny, case_data_schema):
    xls = Dataset(biofabric_tiny, schema=XLSFile)
    # patient_tables = xls.convert(
    #     pz.Table, desc="All tables in the file", cardinality=pz.Cardinality.ONE_TO_MANY)
    patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
    patient_tables = patient_tables.filter("The rows of the table contain the patient age")
    case_data = patient_tables.convert(
        case_data_schema, desc="The patient data in the table", cardinality=Cardinality.ONE_TO_MANY
    )
    return case_data
