import os

import pytest

from palimpzest.corelib import File
from palimpzest.elements import DataRecord


### EXPECTED RECORDS ###
@pytest.fixture
def enron_all_expected_records(enron_eval_tiny_data):
    data_records = []
    for idx, file in enumerate(sorted(os.listdir(enron_eval_tiny_data))):
        dr = DataRecord(schema=File, scan_idx=idx)
        dr.filename = file
        with open(os.path.join(enron_eval_tiny_data, file), "rb") as f:
            dr.contents = f.read()
        data_records.append(dr)

    return data_records


@pytest.fixture
def enron_filter_expected_records(enron_all_expected_records):
    data_records = [
        record
        for record in enron_all_expected_records
        if record.filename
        in [
            "buy-r-inbox-628.txt",
            "buy-r-inbox-749.txt",
            "zipper-a-espeed-28.txt",
        ]
    ]
    return data_records


@pytest.fixture
def real_estate_all_expected_records(real_estate_eval_tiny_data, image_real_estate_listing_schema):
    expected_listings = sorted(os.listdir(real_estate_eval_tiny_data))
    listing_to_modern_and_attractive = {
        "listing1": True,
        "listing2": False,
        "listing3": False,
    }
    listing_to_has_natural_sunlight = {
        "listing1": True,
        "listing2": True,
        "listing3": False,
    }

    data_records = []
    for idx, listing in enumerate(expected_listings):
        dr = DataRecord(schema=image_real_estate_listing_schema, scan_idx=idx)
        dr.listing = listing
        dr.is_modern_and_attractive = listing_to_modern_and_attractive[listing]
        dr.has_natural_sunlight = listing_to_has_natural_sunlight[listing]
        data_records.append(dr)

    return data_records


@pytest.fixture
def real_estate_one_to_many_expected_records(real_estate_eval_tiny_data, room_real_estate_listing_schema):
    expected_listings = sorted(os.listdir(real_estate_eval_tiny_data))
    listing_to_rooms = {
        "listing1": ["other", "living_room", "kitchen"],
        "listing2": ["other", "living_room", "living_room"],
        "listing3": ["other", "living_room", "other"],
    }

    data_records = []
    for idx, listing in enumerate(expected_listings):
        for room in listing_to_rooms[listing]:
            dr = DataRecord(schema=room_real_estate_listing_schema, scan_idx=idx)
            dr.listing = listing
            dr.room = room
            data_records.append(dr)

    return data_records
