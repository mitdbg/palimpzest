import os

import pytest

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import File


### EXPECTED RECORDS ###
@pytest.fixture
def enron_all_expected_records(enron_eval_tiny_data_path):
    data_records = []
    for source_idx, file in enumerate(sorted(os.listdir(enron_eval_tiny_data_path))):
        dr = DataRecord(data_item=File, source_indices=[source_idx])
        dr.filename = file
        with open(os.path.join(enron_eval_tiny_data_path, file), "rb") as f:
            dr.contents = f.read()
        data_records.append(dr)

    return data_records


@pytest.fixture
def enron_filter_expected_records(enron_all_expected_records):
    data_records = [
        record
        for record in enron_all_expected_records
        if record.filename in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]
    ]
    return data_records


@pytest.fixture
def real_estate_all_expected_records(real_estate_eval_tiny_data_path, image_real_estate_listing_schema):
    expected_listings = sorted(os.listdir(real_estate_eval_tiny_data_path))
    listing_to_modern_and_attractive = {"listing1": True, "listing2": False, "listing3": False}
    listing_to_has_natural_sunlight = {"listing1": True, "listing2": True, "listing3": False}

    data_records = []
    for source_idx, listing in enumerate(expected_listings):
        if listing == ".DS_Store":
            continue
        dr = DataRecord(data_item=image_real_estate_listing_schema, source_indices=[source_idx])
        dr.listing = listing
        dr.is_modern_and_attractive = listing_to_modern_and_attractive[listing]
        dr.has_natural_sunlight = listing_to_has_natural_sunlight[listing]
        data_records.append(dr)

    return data_records


@pytest.fixture
def real_estate_one_to_many_expected_records(real_estate_eval_tiny_data_path, room_real_estate_listing_schema):
    expected_listings = sorted(os.listdir(real_estate_eval_tiny_data_path))
    listing_to_rooms = {
        "listing1": ["other", "living_room", "kitchen"],
        "listing2": ["other", "living_room", "living_room"],
        "listing3": ["other", "living_room", "other"],
    }

    data_records = []
    for source_idx, listing in enumerate(expected_listings):
        if listing == ".DS_Store":
            continue
        for room in listing_to_rooms[listing]:
            dr = DataRecord(data_item=room_real_estate_listing_schema, source_indices=[source_idx])
            dr.listing = listing
            dr.room = room
            data_records.append(dr)

    return data_records

# NOTE: this relies on knowledge of the fixtures in fixtures/execution_data.py
@pytest.fixture
def scan_convert_filter_expected_outputs(foobar_schema):
    # create expected outputs to match execution data and champion outputs
    expected_outputs = {}
    for source_idx in range(10):
        if source_idx % 2:
            dr = DataRecord(foobar_schema, [source_idx])
            dr.filename = f"file{source_idx}"
            dr.contents = None
            dr.foo = f"foo{source_idx}"
            dr.bar = f"bar{source_idx}"
            dr._passed_operator = True # bool(source_idx % 2)
            expected_outputs[source_idx] = DataRecordSet([dr], None)

    return expected_outputs

@pytest.fixture
def scan_convert_filter_empty_expected_outputs():
    return {}

@pytest.fixture
def scan_convert_filter_varied_expected_outputs(foobar_schema):
    # create expected outputs to differ from champion outputs;
    # - champion outputs passes odd records
    # - champion outputs always expects bar=f"bar{idx}-{str(Model.GPT_4o)}"
    expected_outputs = {}
    for source_idx in range(10):
        if source_idx % 3 > 0:
            dr = DataRecord(foobar_schema, [source_idx])
            dr.filename = f"file{source_idx}"
            dr.contents = None
            dr.foo = f"foo{source_idx}"
            dr.bar = f"bar{source_idx}-{str(Model.GPT_4o_MINI)}" if source_idx < 6 else f"bar{source_idx}-{str(Model.LLAMA3_1_8B)}"
            dr._passed_operator = True
            expected_outputs[source_idx] = DataRecordSet([dr], None)

    return expected_outputs


@pytest.fixture
def scan_multi_convert_multi_filter_expected_outputs(foobar_schema, baz_schema):
    # create expected outputs to differ from champion outputs;
    # - champion outputs passes source_idx < 5
    # - champion outputs always expects GPT-4 outputs
    # expected outputs:
    # - pass source_idx < 7
    # - always expects GPT-3.5 outputs
    # - does not expect second one-to-many output for source_idx == 0
    expected_outputs = {}
    for source_idx in range(7):
        drs = []
        for one_to_many_idx in range(2):
            if source_idx == 0 and one_to_many_idx == 1:
                continue

            dr = DataRecord(foobar_schema, [source_idx])
            dr.filename = f"file{source_idx}"
            dr.contents = None
            dr.foo = f"foo{source_idx}-one-to-many-{one_to_many_idx}"
            dr.bar = f"bar{source_idx}-{str(Model.GPT_4o_MINI)}"
            dr.baz = f"baz{str(Model.GPT_4o_MINI)}"
            dr._passed_operator = True
            drs.append(dr)

        expected_outputs[source_idx] = DataRecordSet(drs, None)

    return expected_outputs