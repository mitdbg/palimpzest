import pytest


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
    emails = enron_eval_tiny
    emails = emails.sem_add_columns(email_schema)
    emails = emails.sem_filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.sem_filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails


@pytest.fixture
def real_estate_workload(
    real_estate_eval_tiny,
    text_real_estate_listing_schema,
    image_real_estate_listing_schema,
):
    listings = real_estate_eval_tiny
    listings = listings.sem_add_columns(text_real_estate_listing_schema, depends_on="text_content")
    listings = listings.sem_add_columns(image_real_estate_listing_schema, depends_on="image_filepaths")
    listings = listings.sem_filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    listings = listings.filter(
        within_two_miles_of_mit,
        depends_on="address",
    )
    listings = listings.filter(
        in_price_range,
        depends_on="price",
    )
    return listings


@pytest.fixture
def three_converts_workload(enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # construct plan with three converts
    dataset = enron_eval_tiny
    dataset = dataset.sem_add_columns(email_schema)
    dataset = dataset.sem_add_columns(foobar_schema)
    dataset = dataset.sem_add_columns(baz_schema)

    return dataset

@pytest.fixture
def one_filter_one_convert_workload(enron_eval_tiny, email_schema):
    # construct plan with two converts and two filters
    dataset = enron_eval_tiny
    dataset = dataset.sem_filter("filter1")
    dataset = dataset.sem_add_columns(email_schema)

    return dataset

@pytest.fixture
def two_converts_two_filters_workload(enron_eval_tiny, email_schema, foobar_schema):
    # construct plan with two converts and two filters
    dataset = enron_eval_tiny
    dataset = dataset.sem_add_columns(email_schema)
    dataset = dataset.sem_add_columns(foobar_schema)
    dataset = dataset.sem_filter("filter1", depends_on=["sender"])
    dataset = dataset.sem_filter("filter2", depends_on=["subject"])

    return dataset
