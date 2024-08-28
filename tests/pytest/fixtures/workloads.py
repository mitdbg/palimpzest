from io import BytesIO

import pytest
import palimpzest as pz
import pandas as pd

### UDFs ###
def within_two_miles_of_mit(record):
    # NOTE: I'm using this hard-coded function so that folks w/out a
    #       Geocoding API key from google can still run this example
    FAR_AWAY_ADDRS = [
        "Melcher St", "Sleeper St", "437 D St",
        "Seaport Blvd", "50 Liberty Dr", "Telegraph St",
        "Columbia Rd", "E 6th St", "E 7th St", "E 5th St",
    ]
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

### WORKLOADS ###
@pytest.fixture
def enron_workload(enron_eval_tiny, email_schema):
    emails = pz.Dataset(enron_eval_tiny, schema=email_schema)
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
    listings = pz.Dataset(real_estate_eval_tiny, schema=real_estate_listing_files_schema)
    listings = listings.convert(text_real_estate_listing_schema, depends_on="text_content")
    listings = listings.convert(image_real_estate_listing_schema, image_conversion=True, depends_on="image_contents")
    listings = listings.filter(
        "The interior is modern and attractive, and has lots of natural sunlight",
        depends_on=["is_modern_and_attractive", "has_natural_sunlight"],
    )
    listings = listings.filter(within_two_miles_of_mit, depends_on="address")
    listings = listings.filter(in_price_range, depends_on="price")
    return listings

def xls_to_tables(candidate):
    xls_bytes = candidate.contents
    sheet_names = candidate.sheet_names

    records = []
    for sheet_name in sheet_names:
        dataframe = pd.read_excel(
            BytesIO(xls_bytes), sheet_name=sheet_name, engine="openpyxl"
        )

        # TODO extend number of rows with dynamic sizing of context length
        # construct data record
        dr = pz.DataRecord(pz.Table, parent_id=candidate._id)
        rows = []
        for row in dataframe.values[:100]:
            row_record = [str(x) for x in row]
            rows += [row_record]
        dr.rows = rows
        dr.filename = candidate.filename
        dr.header = dataframe.columns.values.tolist()
        dr.name = candidate.filename.split("/")[-1] + "_" + sheet_name
        records.append(dr)

    return records

@pytest.fixture
def biofabric_workload(biofabric_tiny, case_data_schema):
    xls = pz.Dataset(biofabric_tiny, schema=pz.XLSFile)
    # patient_tables = xls.convert(
    #     pz.Table, desc="All tables in the file", cardinality=pz.Cardinality.ONE_TO_MANY)
    patient_tables = xls.convert(pz.Table, udf=lambda record: xls_to_tables(record), cardinality=pz.Cardinality.ONE_TO_MANY)
    patient_tables = patient_tables.filter(
        "The rows of the table contain the patient age"
    )
    case_data = patient_tables.convert(
        case_data_schema, desc="The patient data in the table", cardinality=pz.Cardinality.ONE_TO_MANY
    )
    return case_data
