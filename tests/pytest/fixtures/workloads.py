import pytest
from palimpzest.core.lib.schemas import TextFile
from palimpzest.sets import Dataset


### UDFs ###
def within_two_miles_of_mit(record):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    far_away_addrs = [
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
        return not any([street.lower() in record.address.lower() for street in far_away_addrs])
    except Exception:
        return False


def in_price_range(record):
    try:
        price = record.price
        if isinstance(price, str):
            price = price.strip()
            price = int(price.replace("$", "").replace(",", ""))
        return 6e5 < price <= 2e6
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
    listings = listings.convert(image_real_estate_listing_schema, depends_on="image_filepaths")
    listings = listings.filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    listings = listings.filter(within_two_miles_of_mit, depends_on="address")
    listings = listings.filter(in_price_range, depends_on="price")
    return listings


@pytest.fixture
def three_converts_workload(enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # construct plan with three converts
    dataset = Dataset(enron_eval_tiny, schema=email_schema)
    dataset = dataset.convert(foobar_schema)
    dataset = dataset.convert(baz_schema)

    return dataset

@pytest.fixture
def one_filter_one_convert_workload(enron_eval_tiny, email_schema):
    # construct plan with two converts and two filters
    dataset = Dataset(enron_eval_tiny, schema=TextFile)
    dataset = dataset.filter("filter1")
    dataset = dataset.convert(email_schema)

    return dataset

@pytest.fixture
def two_converts_two_filters_workload(enron_eval_tiny, email_schema, foobar_schema):
    # construct plan with two converts and two filters
    dataset = Dataset(enron_eval_tiny, schema=email_schema)
    dataset = dataset.convert(foobar_schema)
    dataset = dataset.filter("filter1", depends_on=["sender"])
    dataset = dataset.filter("filter2", depends_on=["subject"])

    return dataset
