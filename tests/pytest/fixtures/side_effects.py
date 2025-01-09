import json

import pytest

from palimpzest.core.data.dataclasses import RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet


### Side-Effects for Mocking LLM Calls ###
@pytest.fixture
def enron_filter():
    def mock_call(candidate):
        # determine the answer based on the record filename
        passed_operator = candidate.filename in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]
        
        # create RecordOpStats object with positive time and cost per record
        record_op_stats = RecordOpStats(
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
            record_source_id=candidate._source_id,
            record_state=candidate.as_dict(include_bytes=False),
            op_id="MockFilterFoo",
            logical_op_id="LogicalMockFilterFoo",
            op_name="MockFilter",
            time_per_record=1.0,
            cost_per_record=1.0,
            answer=str(passed_operator),
            passed_operator=passed_operator,
        )

        # set _passed_operator attribute and return
        candidate._passed_operator = passed_operator

        return DataRecordSet([candidate], [record_op_stats])

    return mock_call


@pytest.fixture
def enron_convert(email_schema):
    def mock_call(candidate):
        filename_to_sender = {
            "buy-r-inbox-628.txt": "sherron.watkins@enron.com",
            "buy-r-inbox-749.txt": "david.port@enron.com",
            "kaminski-v-deleted-items-1902.txt": "vkaminski@aol.com",
            "martin-t-inbox-96-short.txt": "sarah.palmer@enron.com",
            "skilling-j-inbox-1109.txt": "gary@cioclub.com",
            "zipper-a-espeed-28.txt": "travis.mccullough@enron.com",
        }
        filename_to_subject = {
            "buy-r-inbox-628.txt": "RE: portrac",
            "buy-r-inbox-749.txt": "RE: NewPower",
            "kaminski-v-deleted-items-1902.txt": "Fwd: FYI",
            "martin-t-inbox-96-short.txt": "Enron Mentions -- 01/18/02",
            "skilling-j-inbox-1109.txt": "Information Security Executive -092501",
            "zipper-a-espeed-28.txt": "Redraft of the Exclusivity Agreement",
        }

        # construct data record
        dr = DataRecord.from_parent(schema=email_schema, parent_record=candidate, cardinality_idx=0)
        dr.sender = filename_to_sender[candidate.filename]
        dr.subject = filename_to_subject[candidate.filename]
        dr.filename = candidate.filename
        dr.contents = candidate.contents

        # compute fake record_op_stats
        record_op_stats = RecordOpStats(
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
            record_source_id=candidate._source_id,
            record_state=dr.as_dict(include_bytes=False),
            op_id="MockConvertFoo",
            logical_op_id="LogicalMockConvertFoo",
            op_name="MockConvert",
            time_per_record=1.0,
            cost_per_record=1.0,
            answer=json.dumps({"sender": dr.sender, "subject": dr.subject}),
        )

        return DataRecordSet([dr], [record_op_stats])

    return mock_call


@pytest.fixture
def real_estate_convert(image_real_estate_listing_schema):
    def mock_call(candidate):
        listing_to_modern_and_attractive = {"listing1": True, "listing2": False, "listing3": False}
        listing_to_has_natural_sunlight = {"listing1": True, "listing2": True, "listing3": False}

        # construct data record
        dr = DataRecord.from_parent(schema=image_real_estate_listing_schema, parent_record=candidate, cardinality_idx=0)
        dr.is_modern_and_attractive = listing_to_modern_and_attractive[candidate.listing]
        dr.has_natural_sunlight = listing_to_has_natural_sunlight[candidate.listing]
        dr.listing = candidate.listing
        dr.text_content = candidate.text_content
        dr.image_filepaths = candidate.image_filepaths

        # compute fake record_op_stats
        record_op_stats = RecordOpStats(
            record_id=candidate._id,
            record_parent_id=candidate._parent_id,
            record_source_id=candidate._source_id,
            record_state=dr.as_dict(include_bytes=False),
            op_id="MockConvertFoo",
            logical_op_id="LogicalMockConvertFoo",
            op_name="MockConvert",
            time_per_record=1.0,
            cost_per_record=1.0,
            answer=json.dumps(
                {
                    "is_modern_and_attractive": dr.is_modern_and_attractive,
                    "has_natural_sunlight": dr.has_natural_sunlight,
                }
            ),
        )

        return DataRecordSet([dr], [record_op_stats])

    return mock_call


@pytest.fixture
def real_estate_one_to_many_convert(room_real_estate_listing_schema):
    def mock_call(candidate):
        listing_to_rooms = {
            "listing1": ["other", "living_room", "kitchen"],
            "listing2": ["other", "living_room", "living_room"],
            "listing3": ["other", "living_room", "other"],
        }

        # construct data records and list of RecordOpStats
        data_records, record_op_stats_lst = [], []
        for idx, room in enumerate(listing_to_rooms[candidate.listing]):
            # create data record
            dr = DataRecord.from_parent(schema=room_real_estate_listing_schema, parent_record=candidate, cardinality_idx=idx)
            dr.room = room
            dr.listing = candidate.listing
            dr.text_content = candidate.text_content
            dr.image_filepaths = candidate.image_filepaths
            data_records.append(dr)

            # create fake record_op_stats
            record_op_stats = RecordOpStats(
                record_id=candidate._id,
                record_parent_id=candidate._parent_id,
                record_source_id=candidate._source_id,
                record_state=dr.as_dict(include_bytes=False),
                op_id="MockConvertFoo",
                logical_op_id="LogicalMockConvertFoo",
                op_name="MockConvert",
                time_per_record=1.0,
                cost_per_record=1.0,
                answer=json.dumps({"room": listing_to_rooms[candidate.listing]}),
            )
            record_op_stats_lst.append(record_op_stats)

        return DataRecordSet(data_records, record_op_stats_lst)

    return mock_call
