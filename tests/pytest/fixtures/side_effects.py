import pytest

from palimpzest.core.data.dataclasses import GenerationStats


### Side-Effects for Mocking LLM Calls ###
@pytest.fixture
def enron_filter():
    def mock_filter(candidate):
        # determine the answer based on the record filename
        field_answers = {"passed_operator": candidate.filename in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]}
        generation_stats = GenerationStats(cost_per_record=1.0)

        return field_answers, generation_stats

    return mock_filter


@pytest.fixture
def enron_convert(email_schema):
    def mock_convert(candidate, fields):
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

        # determine the answer based on the record filename
        field_answers = {
            "sender": [filename_to_sender[candidate.filename]],
            "subject": [filename_to_subject[candidate.filename]],
        }
        generation_stats = GenerationStats(cost_per_record=1.0)

        return field_answers, generation_stats

    return mock_convert


@pytest.fixture
def real_estate_convert(image_real_estate_listing_schema):
    def mock_convert(candidate, fields):
        listing_to_modern_and_attractive = {"listing1": True, "listing2": False, "listing3": False}
        listing_to_has_natural_sunlight = {"listing1": True, "listing2": True, "listing3": False}

        # determine the answer based on the record listing
        field_answers = {
            "is_modern_and_attractive": [listing_to_modern_and_attractive[candidate.listing]],
            "has_natural_sunlight": [listing_to_has_natural_sunlight[candidate.listing]],
        }
        generation_stats = GenerationStats(cost_per_record=1.0)

        return field_answers, generation_stats

    return mock_convert


@pytest.fixture
def real_estate_one_to_many_convert(room_real_estate_listing_schema):
    def mock_convert(candidate, fields):
        listing_to_rooms = {
            "listing1": ["other", "living_room", "kitchen"],
            "listing2": ["other", "living_room", "living_room"],
            "listing3": ["other", "living_room", "other"],
        }

        # determine the answers based on the record listing
        field_answers = {"room": listing_to_rooms[candidate.listing]}
        generation_stats = GenerationStats(cost_per_record=1.0)

        return field_answers, generation_stats

    return mock_convert
