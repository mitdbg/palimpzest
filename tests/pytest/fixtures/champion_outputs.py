import pytest

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import TextFile


# NOTE: this relies on knowledge of the fixtures in fixtures/execution_data.py
@pytest.fixture
def scan_convert_filter_champion_outputs(scan_convert_filter_sentinel_plan, foobar_schema):
    logical_op_ids = scan_convert_filter_sentinel_plan.logical_op_ids
    scan_logical_op_id = logical_op_ids[0]
    convert_logical_op_id = logical_op_ids[1]
    filter_logical_op_id = logical_op_ids[2]
    champion_outputs = {logical_op_id: {} for logical_op_id in logical_op_ids}

    # compute scan champion_outputs
    for source_idx in range(10):
        scan_dr = DataRecord(TextFile, [source_idx], parent_ids=None)
        scan_dr.filename = f"file{source_idx}"
        scan_dr.contents = None
        champion_outputs[scan_logical_op_id][source_idx] = DataRecordSet([scan_dr], None)

    # add convert champion outputs
    for source_idx in range(10):
        convert_dr = DataRecord(foobar_schema, [source_idx])
        convert_dr.filename = f"file{source_idx}"
        convert_dr.contents = None
        convert_dr.foo = f"foo{source_idx}"
        convert_dr.bar = f"bar{source_idx}"
        champion_outputs[convert_logical_op_id][source_idx] = DataRecordSet([convert_dr], None)

    # add filter champion outputs
    for source_idx in range(10):
        filter_dr = DataRecord(foobar_schema, [source_idx])
        filter_dr.filename = f"file{source_idx}"
        filter_dr.contents = None
        filter_dr.foo = f"foo{source_idx}"
        filter_dr.bar = f"bar{source_idx}"
        filter_dr.passed_operator = bool(source_idx % 2)
        champion_outputs[filter_logical_op_id][source_idx] = DataRecordSet([filter_dr], None)

    return champion_outputs


@pytest.fixture
def scan_convert_filter_empty_champion_outputs(scan_convert_filter_sentinel_plan, foobar_schema):
    logical_op_ids = scan_convert_filter_sentinel_plan.logical_op_ids
    scan_logical_op_id = logical_op_ids[0]
    convert_logical_op_id = logical_op_ids[1]
    filter_logical_op_id = logical_op_ids[2]
    champion_outputs = {logical_op_id: {} for logical_op_id in logical_op_ids}

    # compute scan champion_outputs
    for source_idx in range(10):
        scan_dr = DataRecord(TextFile, [source_idx], parent_ids=None)
        scan_dr.filename = f"file{source_idx}"
        scan_dr.contents = None
        champion_outputs[scan_logical_op_id][source_idx] = DataRecordSet([scan_dr], None)

    # add convert champion outputs
    for source_idx in range(10):
        convert_dr = DataRecord(foobar_schema, [source_idx])
        convert_dr.filename = f"file{source_idx}"
        convert_dr.contents = None
        convert_dr.foo = f"foo{source_idx}"
        convert_dr.bar = f"bar{source_idx}"
        champion_outputs[convert_logical_op_id][source_idx] = DataRecordSet([convert_dr], None)

    # add filter champion outputs
    for source_idx in range(10):
        filter_dr = DataRecord(foobar_schema, [source_idx])
        filter_dr.filename = f"file{source_idx}"
        filter_dr.contents = None
        filter_dr.foo = f"foo{source_idx}"
        filter_dr.bar = f"bar{source_idx}"
        filter_dr.passed_operator = False
        champion_outputs[filter_logical_op_id][source_idx] = DataRecordSet([filter_dr], None)

    return champion_outputs


@pytest.fixture
def scan_convert_filter_varied_champion_outputs(scan_convert_filter_sentinel_plan, foobar_schema):
    logical_op_ids = scan_convert_filter_sentinel_plan.logical_op_ids
    scan_logical_op_id = logical_op_ids[0]
    convert_logical_op_id = logical_op_ids[1]
    filter_logical_op_id = logical_op_ids[2]
    champion_outputs = {logical_op_id: {} for logical_op_id in logical_op_ids}

    # compute scan champion_outputs
    for source_idx in range(10):
        scan_dr = DataRecord(TextFile, [source_idx], parent_ids=None)
        scan_dr.filename = f"file{source_idx}"
        scan_dr.contents = None
        champion_outputs[scan_logical_op_id][source_idx] = DataRecordSet([scan_dr], None)

    # add convert champion outputs
    for source_idx in range(10):
        convert_dr = DataRecord(foobar_schema, [source_idx])
        convert_dr.filename = f"file{source_idx}"
        convert_dr.contents = None
        convert_dr.foo = f"foo{source_idx}"
        convert_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
        champion_outputs[convert_logical_op_id][source_idx] = DataRecordSet([convert_dr], None)

    # add filter champion outputs
    for source_idx in range(10):
        filter_dr = DataRecord(foobar_schema, [source_idx])
        filter_dr.filename = f"file{source_idx}"
        filter_dr.contents = None
        filter_dr.foo = f"foo{source_idx}"
        filter_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
        filter_dr.passed_operator = bool(source_idx % 2)
        champion_outputs[filter_logical_op_id][source_idx] = DataRecordSet([filter_dr], None)

    return champion_outputs


@pytest.fixture
def scan_multi_convert_multi_filter_champion_outputs(scan_multi_convert_multi_filter_sentinel_plan, foobar_schema, baz_schema):
    """
    Champion outputs agree with GPT-4.
    """
    logical_op_ids = scan_multi_convert_multi_filter_sentinel_plan.logical_op_ids
    scan_logical_op_id = logical_op_ids[0]
    convert1_logical_op_id = logical_op_ids[1]
    filter1_logical_op_id = logical_op_ids[2]
    filter2_logical_op_id = logical_op_ids[3]
    convert2_logical_op_id = logical_op_ids[4]
    champion_outputs = {logical_op_id: {} for logical_op_id in logical_op_ids}

    # compute scan champion_outputs
    for source_idx in range(10):
        scan_dr = DataRecord(TextFile, [source_idx], parent_ids=None)
        scan_dr.filename = f"file{source_idx}"
        scan_dr.contents = None
        champion_outputs[scan_logical_op_id][source_idx] = DataRecordSet([scan_dr], None)

    # add first convert champion outputs
    for source_idx in range(10):
        drs = []
        for one_to_many_idx in range(2):
            convert_dr = DataRecord(foobar_schema, [source_idx])
            convert_dr.filename = f"file{source_idx}"
            convert_dr.contents = None
            convert_dr.foo = f"foo{source_idx}-one-to-many-{one_to_many_idx}"
            convert_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
            drs.append(convert_dr)

        champion_outputs[convert1_logical_op_id][source_idx] = DataRecordSet(drs, None)

    # add first filter champion outputs
    for source_idx in range(10):
        for one_to_many_idx in range(2):
            filter_dr = DataRecord(foobar_schema, [source_idx])
            filter_dr.filename = f"file{source_idx}"
            filter_dr.contents = None
            filter_dr.foo = f"foo{source_idx}-one-to-many-{one_to_many_idx}"
            filter_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
            filter_dr.passed_operator = bool(source_idx < 7)
            champion_outputs[filter1_logical_op_id][source_idx] = DataRecordSet([filter_dr], None)

    # add second filter champion outputs
    for source_idx in range(7):
        for one_to_many_idx in range(2):
            filter_dr = DataRecord(foobar_schema, [source_idx])
            filter_dr.filename = f"file{source_idx}"
            filter_dr.contents = None
            filter_dr.foo = f"foo{source_idx}-one-to-many-{one_to_many_idx}"
            filter_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
            filter_dr.passed_operator = bool(source_idx < 5)
            champion_outputs[filter2_logical_op_id][source_idx] = DataRecordSet([filter_dr], None)

    # add first convert champion outputs
    for source_idx in range(5):
        for one_to_many_idx in range(2):
            convert_dr = DataRecord(baz_schema, [source_idx])
            convert_dr.filename = f"file{source_idx}"
            convert_dr.contents = None
            convert_dr.foo = f"foo{source_idx}-one-to-many-{one_to_many_idx}"
            convert_dr.bar = f"bar{source_idx}-{str(Model.GPT_4o)}"
            convert_dr.baz = f"baz{str(Model.GPT_4o)}"
            champion_outputs[convert2_logical_op_id][source_idx] = DataRecordSet([convert_dr], None)

    return champion_outputs
