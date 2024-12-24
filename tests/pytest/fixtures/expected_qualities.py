import re

import pytest

from palimpzest.constants import Model


# NOTE: this relies on knowledge of the fixtures in fixtures/execution_data.py
@pytest.fixture
def scan_convert_filter_qualities(scan_convert_filter_execution_data):
    expected_qualities = {
        op_set_id: {
            source_id: [[1.0] for _ in record_sets]
            for source_id, record_sets in source_id_to_record_sets.items()
        }
        for op_set_id, source_id_to_record_sets in scan_convert_filter_execution_data.items()
    }
    return expected_qualities

@pytest.fixture
def scan_convert_filter_empty_qualities(scan_convert_filter_execution_data):
    expected_qualities = {}
    for op_set_id, source_id_to_record_sets in scan_convert_filter_execution_data.items():
        expected_qualities[op_set_id] = {}
        for source_id, record_sets in source_id_to_record_sets.items():
            expected_qualities[op_set_id][source_id] = []
            for record_set in record_sets:
                record_set_expected_qualities = []
                for record_op_stats in record_set.record_op_stats:
                    quality = None
                    if record_op_stats.logical_op_id == "scan1-logical":  # noqa: SIM114
                        quality = 1.0
                    elif record_op_stats.logical_op_id == "convert1-logical":
                        quality = 1.0
                    elif record_op_stats.logical_op_id == "filter1-logical":
                        # by construction, champion model expects no outputs but models output odd records,
                        # so odd records get quality 0.0 and even records get quality 1.0
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        quality = int(not bool(source_idx % 2))

                    record_set_expected_qualities.append(quality)
                expected_qualities[op_set_id][source_id].append(record_set_expected_qualities)

    return expected_qualities

@pytest.fixture
def scan_convert_filter_varied_qualities(scan_convert_filter_varied_execution_data):
    expected_qualities = {}
    for op_set_id, source_id_to_record_sets in scan_convert_filter_varied_execution_data.items():
        expected_qualities[op_set_id] = {}
        for source_id, record_sets in source_id_to_record_sets.items():
            expected_qualities[op_set_id][source_id] = []
            for record_set in record_sets:
                record_set_expected_qualities = []
                for record_op_stats in record_set.record_op_stats:
                    quality = None
                    if record_op_stats.logical_op_id == "scan1-logical":
                        quality = 1.0
                    elif record_op_stats.logical_op_id == "convert1-logical":
                        quality = 1.0 if str(Model.GPT_4o) in record_op_stats.op_id else 0.5
                    elif record_op_stats.logical_op_id == "filter1-logical":
                        if str(Model.GPT_4o) in record_op_stats.op_id:
                            quality = 1.0
                        elif str(Model.GPT_4o_MINI) in record_op_stats.op_id:
                            # by construction, champion model expects odd record outputs but GPT-3.5 outputs even records,
                            # so all records get quality 0.0
                            quality = 0.0
                        elif str(Model.MIXTRAL) in record_op_stats.op_id:
                            # by construction, champion model expects odd record outputs but Mixtral outputs all records,
                            # so even records get quality 0.0 and odd records get quality 1.0
                            match = re.match(r"source(.+?)", source_id)
                            source_idx = int(match.group(1))
                            quality = int(bool(source_idx % 2))

                    record_set_expected_qualities.append(quality)
                expected_qualities[op_set_id][source_id].append(record_set_expected_qualities)

    return expected_qualities

@pytest.fixture
def scan_convert_filter_varied_override_qualities(scan_convert_filter_varied_execution_data):
    """
    NOTE: this test in particular kind of sucks, it is really hard to verify what correct behavior is

    The score_quality() function will use expected_output record for scoring quality when (a subset
    of) its record state matches a record_op_stats object perfectly. If no match is found, then the
    champion model is used. Qualities here are computed accordingly.
    """
    expected_qualities = {}
    for op_set_id, source_id_to_record_sets in scan_convert_filter_varied_execution_data.items():
        expected_qualities[op_set_id] = {}
        for source_id, record_sets in source_id_to_record_sets.items():
            expected_qualities[op_set_id][source_id] = []
            for record_set in record_sets:
                record_set_expected_qualities = []
                for record_op_stats in record_set.record_op_stats:
                    quality = None
                    if record_op_stats.logical_op_id == "scan1-logical":
                        quality = 1.0
                    elif record_op_stats.logical_op_id == "convert1-logical":
                        # by construction, expected output is used to score records with idx % 3 > 0, i.e. records 1, 2, 4, 5, 7, 8
                        # for expected outputs w/record idx < 6 (i.e. records 1, 2, 4, 5) the expected `bar` value is f"bar{idx}-{str(Model.GPT_4o_MINI)}";
                        # for expected outputs w/record idx >= 6 (i.e. records 7, 8) the expected `bar` value is f"bar{idx}-{str(Model.MIXTRAL)}";
                        # for records 0, 3, 6, 9; the champion model expects outputs f"bar{idx}-{str(Model.GPT_4o)}"
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        if source_idx % 3 > 0 and source_idx < 6:
                            quality = 1.0 if str(Model.GPT_4o_MINI) in record_op_stats.op_id else 0.5
                        elif source_idx % 3 > 0:
                            quality = 1.0 if str(Model.MIXTRAL) in record_op_stats.op_id else 0.5
                        else:
                            quality = 1.0 if str(Model.GPT_4o) in record_op_stats.op_id else 0.5

                    elif record_op_stats.logical_op_id == "filter1-logical":
                        # by construction, expected output passes all records with idx % 3 > 0, i.e. records 1, 2, 4, 5, 7, 8
                        # - it expects GPT-3.5 for records with idx < 6 (i.e. records 1, 2, 4, 5)
                        # - it expects MIXTRAL for records with idx >= 6 (i.e. records 7, 8)
                        # champion model passes all odd records
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))

                        # using expected_record and match found
                        if source_idx % 3 > 0 and source_idx < 6 and str(Model.GPT_4o_MINI) in record_op_stats.op_id:  # noqa: SIM114
                            quality = int(record_op_stats.passed_operator)
                        
                        elif source_idx % 3 > 0 and source_idx >= 6 and str(Model.MIXTRAL) in record_op_stats.op_id:  # noqa: SIM114
                            quality = int(record_op_stats.passed_operator)

                        # using champion record and it thinks record should pass
                        elif source_idx % 2:
                            quality = int(record_op_stats.passed_operator)

                        # using champion record and it thinks record should not pass
                        else:
                            quality = int(not record_op_stats.passed_operator)

                    record_set_expected_qualities.append(quality)
                expected_qualities[op_set_id][source_id].append(record_set_expected_qualities)

    return expected_qualities


@pytest.fixture
def scan_multi_convert_multi_filter_qualities(scan_multi_convert_multi_filter_execution_data):
    expected_qualities = {}
    for op_set_id, source_id_to_record_sets in scan_multi_convert_multi_filter_execution_data.items():
        expected_qualities[op_set_id] = {}
        for source_id, record_sets in source_id_to_record_sets.items():
            expected_qualities[op_set_id][source_id] = []
            for record_set in record_sets:
                record_set_expected_qualities = []
                for one_to_many_idx, record_op_stats in enumerate(record_set.record_op_stats):
                    quality = None
                    if record_op_stats.logical_op_id == "scan1-logical":
                        quality = 1.0
                    elif record_op_stats.logical_op_id == "convert1-logical":
                        # by construction, expected output is used to score records with source_idx < 7
                        # the second output (one_to_many_idx == 1) for source_idx == 0 is not expected
                        # the expected `bar` value is f"bar{source_idx}-{str(Model.GPT_4o_MINI)}";
                        # for records with source_idx >= 7; the champion model expects outputs f"bar{idx}-{str(Model.GPT_4o)}"
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        if source_idx == 0 and one_to_many_idx == 1:
                            quality = 0.0
                        elif source_idx < 7:
                            quality = 1.0 if str(Model.GPT_4o_MINI) in record_op_stats.op_id else 0.5
                        else:
                            quality = 1.0 if str(Model.GPT_4o) in record_op_stats.op_id else 0.5

                    elif record_op_stats.logical_op_id == "filter1-logical":
                        # by construction, expected output passes all records with source_idx < 7
                        # champion model also passes all records with source_idx < 7
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        if source_idx < 7:
                            quality = int(record_op_stats.passed_operator)
                        else:
                            quality = int(not record_op_stats.passed_operator)

                    elif record_op_stats.logical_op_id == "filter2-logical":
                        # by construction, expected output passes all records with source_idx < 7
                        # champion model passes all records with source_idx < 5
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        
                        # the champion model and expected output agree on the filter decision for all records with source_idx < 5
                        if source_idx < 5:  # noqa: SIM114
                            quality = int(record_op_stats.passed_operator)

                        # for records with source_ids in [5, 6] if the *convert& model used was GPT_3_5, then the
                        # expected record will match and we expect the record to pass
                        elif source_idx < 7 and str(Model.GPT_4o_MINI) in record_op_stats.record_state['bar']:
                            quality = int(record_op_stats.passed_operator)

                        # for records with source_ids in [5, 6] if the model used was *not* GPT_3_5, then the champion model will be used
                        # and it does *not* expect the record to pass
                        elif source_idx < 7:
                            quality = int(not record_op_stats.passed_operator)

                        # for all records with source_ids >= 7, the champion is used and it does not pass the record
                        else:
                            quality = int(not record_op_stats.passed_operator)

                    elif record_op_stats.logical_op_id == "convert2-logical":
                        # by construction, expected output is used to score records with source_idx < 7
                        # the second output (one_to_many_idx == 1) for source_idx == 0 is not expected
                        # the expected `bar` value is f"bar{source_idx}-{str(Model.GPT_4o_MINI)}";
                        # for records with source_idx >= 7; the champion model expects outputs f"bar{idx}-{str(Model.GPT_4o)}"
                        match = re.match(r"source(.+?)", source_id)
                        source_idx = int(match.group(1))
                        if source_idx == 0 and one_to_many_idx == 1:
                            quality = 0.0
                        else:
                            quality = 1.0 if str(Model.GPT_4o_MINI) in record_op_stats.op_id else 0.0

                    record_set_expected_qualities.append(quality)
                expected_qualities[op_set_id][source_id].append(record_set_expected_qualities)

    return expected_qualities
