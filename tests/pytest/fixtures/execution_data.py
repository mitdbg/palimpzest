from palimpzest.constants import Model
from palimpzest.corelib import TextFile
from palimpzest.elements import DataRecord, DataRecordSet
from palimpzest.dataclasses import RecordOpStats
from palimpzest.optimizer import SentinelPlan
import pytest
import re


# NOTE: technically the filter should process 10 outputs 3x times
@pytest.fixture
def scan_convert_filter_execution_data(scan_convert_filter_sentinel_plan, foobar_schema):
    # initialize execution data
    op_sets = scan_convert_filter_sentinel_plan.operator_sets
    op_set_ids = [SentinelPlan.compute_op_set_id(op_set) for op_set in op_sets]
    scan_op_set_id = op_set_ids[0]
    convert_op_set_id = op_set_ids[1]
    filter_op_set_id = op_set_ids[2]
    execution_data = {op_set_id: {} for op_set_id in op_set_ids}

    # create data records first
    scan_drs, convert_drs, filter_drs = [], [], []
    for idx in range(10):
        source_id = f"source{idx}"
        scan_dr = DataRecord(TextFile, source_id, parent_id=None)
        scan_dr.filename = f"file{idx}"
        scan_dr.contents = None
        scan_drs.append(scan_dr)

    # create convert data records
    for idx in range(30):
        convert_dr = DataRecord.fromParent(foobar_schema, scan_drs[idx % 10])
        convert_dr.foo = f"foo{idx % 10}"
        convert_dr.bar = f"bar{idx % 10}"
        convert_drs.append(convert_dr)

    # create filter data records
    for idx in range(30):
        filter_dr = DataRecord.fromParent(foobar_schema, convert_drs[idx])
        filter_drs.append(filter_dr)

    # create execution data entries for scan operator
    for scan_dr in scan_drs:
        op_id = op_sets[0][0].get_op_id()
        source_id = scan_dr._source_id
        record_op_stats = RecordOpStats(
            record_id=scan_dr._id,
            record_parent_id=scan_dr._parent_id,
            record_source_id=scan_dr._source_id,
            op_id=op_id,
            op_name="MarshalAndScanDataOp",
            time_per_record=1.0,
            cost_per_record=0.0,
            logical_op_id=f"scan1-logical",
            record_state=scan_dr._asDict(),
            passed_filter=None,
            generated_fields=None,
        )
        scan_record_set = DataRecordSet([scan_dr], [record_op_stats])
        execution_data[scan_op_set_id][source_id] = [scan_record_set]

    # create execution data entries for convert operator
    for op_idx, op in enumerate(op_sets[1]):
        op_id = op.get_op_id()
        for source_idx in range(10):
            record_idx = op_idx * len(op_sets) + source_idx
            convert_dr = convert_drs[record_idx]
            source_id = convert_dr._source_id
            record_op_stats = RecordOpStats(
                record_id=convert_dr._id,
                record_parent_id=convert_dr._parent_id,
                record_source_id=convert_dr._source_id,
                op_id=op_id,
                op_name="LLMConvertBonded",
                time_per_record=1.0,
                cost_per_record=1.0,
                logical_op_id=f"convert1-logical",
                record_state=convert_dr._asDict(),
                passed_filter=None,
                generated_fields=["foo", "bar"],
            )
            convert_record_set = DataRecordSet([convert_dr], [record_op_stats])
            if source_id not in execution_data[convert_op_set_id]:
                execution_data[convert_op_set_id][source_id] = [convert_record_set]
            else:
                execution_data[convert_op_set_id][source_id].append(convert_record_set)

    # create execution data entries for filter operator
    for op_idx, op in enumerate(op_sets[2]):
        op_id = op.get_op_id()
        for source_idx in range(10):
            record_idx = op_idx * len(op_sets) + source_idx
            filter_dr = filter_drs[record_idx]
            source_id = filter_dr._source_id
            match = re.match(r"source(.+?)", source_id)
            source_idx = int(match.group(1))
            record_op_stats = RecordOpStats(
                record_id=filter_dr._id,
                record_parent_id=filter_dr._parent_id,
                record_source_id=filter_dr._source_id,
                op_id=op_id,
                op_name="LLMFilter",
                time_per_record=1.0,
                cost_per_record=1.0,
                logical_op_id=f"filter1-logical",
                record_state=filter_dr._asDict(),
                passed_filter=bool(source_idx % 2), # odd examples pass filter
                generated_fields=None,
            )
            filter_record_set = DataRecordSet([filter_dr], [record_op_stats])
            if source_id not in execution_data[filter_op_set_id]:
                execution_data[filter_op_set_id][source_id] = [filter_record_set]
            else:
                execution_data[filter_op_set_id][source_id].append(filter_record_set)

    return execution_data


# NOTE: technically the filter should process 10 outputs 3x times
@pytest.fixture
def scan_convert_filter_varied_execution_data(scan_convert_filter_sentinel_plan, foobar_schema):
    # initialize execution data
    op_sets = scan_convert_filter_sentinel_plan.operator_sets
    op_set_ids = [SentinelPlan.compute_op_set_id(op_set) for op_set in op_sets]
    scan_op_set_id = op_set_ids[0]
    convert_op_set_id = op_set_ids[1]
    filter_op_set_id = op_set_ids[2]
    execution_data = {op_set_id: {} for op_set_id in op_set_ids}

    # create data records first
    scan_drs, convert_drs, filter_drs = [], [], []
    for idx in range(10):
        source_id = f"source{idx}"
        scan_dr = DataRecord(TextFile, source_id, parent_id=None)
        scan_dr.filename = f"file{idx}"
        scan_dr.contents = None
        scan_drs.append(scan_dr)

    # create convert data records
    models = [Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL]
    for model in models:
        for idx in range(10):
            convert_dr = DataRecord.fromParent(foobar_schema, scan_drs[idx])
            convert_dr.foo = f"foo{idx}"
            convert_dr.bar = f"bar{idx}-{str(model)}"
            convert_drs.append(convert_dr)

    # create filter data records
    for idx in range(30):
        filter_dr = DataRecord.fromParent(foobar_schema, convert_drs[idx])
        filter_drs.append(filter_dr)

    # create execution data entries for scan operator
    for scan_dr in scan_drs:
        source_id = scan_dr._source_id
        record_op_stats = RecordOpStats(
            record_id=scan_dr._id,
            record_parent_id=scan_dr._parent_id,
            record_source_id=scan_dr._source_id,
            op_id=f"scan1-phys",
            op_name="MarshalAndScanDataOp",
            time_per_record=1.0,
            cost_per_record=0.0,
            logical_op_id=f"scan1-logical",
            record_state=scan_dr._asDict(),
            passed_filter=None,
            generated_fields=None,
        )
        scan_record_set = DataRecordSet([scan_dr], [record_op_stats])
        execution_data[scan_op_set_id][source_id] = [scan_record_set]

    # create execution data entries for convert operator
    for idx, convert_dr in enumerate(convert_drs):
        source_id = convert_dr._source_id
        model = models[idx // 10]
        record_op_stats = RecordOpStats(
            record_id=convert_dr._id,
            record_parent_id=convert_dr._parent_id,
            record_source_id=convert_dr._source_id,
            op_id=f"convert1-phys-{str(model)}",
            op_name="LLMConvertBonded",
            time_per_record=1.0,
            cost_per_record=1.0,
            logical_op_id=f"convert1-logical",
            record_state=convert_dr._asDict(),
            passed_filter=None,
            generated_fields=["foo", "bar"],
        )
        convert_record_set = DataRecordSet([convert_dr], [record_op_stats])
        if source_id not in execution_data[convert_op_set_id]:
            execution_data[convert_op_set_id][source_id] = [convert_record_set]
        else:
            execution_data[convert_op_set_id][source_id].append(convert_record_set)

    # create execution data entries for filter operator
    for idx, filter_dr in enumerate(filter_drs):
        source_id = filter_dr._source_id
        model = models[idx // 10]
        source_idx = idx % 10

        # GPT-4 passes odd examples
        # GPT-3.5 passes even examples
        # Mixtral passes all examples
        passed_filter = None
        if model == Model.GPT_4o:
            passed_filter = bool(source_idx % 2)
        elif model == Model.GPT_4o_MINI:
            passed_filter = not bool(source_idx % 2)
        elif model == Model.MIXTRAL:
            passed_filter = True

        record_op_stats = RecordOpStats(
            record_id=filter_dr._id,
            record_parent_id=filter_dr._parent_id,
            record_source_id=filter_dr._source_id,
            op_id=f"filter1-phys-{str(model)}",
            op_name="LLMFilter",
            time_per_record=1.0,
            cost_per_record=1.0,
            logical_op_id=f"filter1-logical",
            record_state=filter_dr._asDict(),
            passed_filter=passed_filter,
            generated_fields=None,
        )
        filter_record_set = DataRecordSet([filter_dr], [record_op_stats])
        if source_id not in execution_data[filter_op_set_id]:
            execution_data[filter_op_set_id][source_id] = [filter_record_set]
        else:
            execution_data[filter_op_set_id][source_id].append(filter_record_set)

    return execution_data


@pytest.fixture
def scan_multi_convert_multi_filter_execution_data(scan_multi_convert_multi_filter_sentinel_plan, foobar_schema, baz_schema):
    # initialize execution data
    op_sets = scan_multi_convert_multi_filter_sentinel_plan.operator_sets
    op_set_ids = [SentinelPlan.compute_op_set_id(op_set) for op_set in op_sets]
    scan_op_set_id = op_set_ids[0]
    convert1_op_set_id = op_set_ids[1]
    filter1_op_set_id = op_set_ids[2]
    filter2_op_set_id = op_set_ids[3]
    convert2_op_set_id = op_set_ids[4]
    execution_data = {op_set_id: {} for op_set_id in op_set_ids}

    # create data records first
    scan_drs, convert1_drs, convert2_drs, filter1_drs, filter2_drs = [], [], [], [], []
    for idx in range(10):
        source_id = f"source{idx}"
        scan_dr = DataRecord(TextFile, source_id, parent_id=None)
        scan_dr.filename = f"file{idx}"
        scan_dr.contents = None
        scan_drs.append(scan_dr)

    # create first convert data records
    models = [Model.GPT_4o, Model.GPT_4o_MINI, Model.MIXTRAL]
    for model in models:
        for idx in range(10):
            for one_to_many_idx in range(2):
                convert_dr = DataRecord.fromParent(foobar_schema, scan_drs[idx])
                convert_dr.foo = f"foo{idx}-one-to-many-{one_to_many_idx}"
                convert_dr.bar = f"bar{idx}-{str(model)}"
                convert1_drs.append(convert_dr)

    # create first filter data records
    for model in models:
        for gpt4_convert_dr in convert1_drs[:20]:
            filter_dr = DataRecord.fromParent(foobar_schema, gpt4_convert_dr)
            filter1_drs.append(filter_dr)

    # NOTE: assume GPT-4 in filter1 filtered out last 6 out of 20 records
    # create second filter data records
    for model in models:
        for gpt4_filter_dr in filter1_drs[:14]:
            filter_dr = DataRecord.fromParent(foobar_schema, gpt4_filter_dr)
            filter2_drs.append(filter_dr)

    # NOTE: assume GPT-4 in filter2 filtered out last 4 out of 14 records
    # create second convert data records (second half of records will be filtered out)
    for model in models:
        for gpt4_filter_dr in filter2_drs[:10]:
            convert_dr = DataRecord.fromParent(baz_schema, gpt4_filter_dr)
            convert_dr.baz = f"baz{str(model)}"
            convert2_drs.append(convert_dr)

    # create execution data entries for scan operator
    for scan_dr in scan_drs:
        source_id = scan_dr._source_id
        record_op_stats = RecordOpStats(
            record_id=scan_dr._id,
            record_parent_id=scan_dr._parent_id,
            record_source_id=scan_dr._source_id,
            op_id=f"scan1-phys",
            op_name="MarshalAndScanDataOp",
            time_per_record=1.0,
            cost_per_record=0.0,
            logical_op_id=f"scan1-logical",
            record_state=scan_dr._asDict(),
            passed_filter=None,
            generated_fields=None,
        )
        scan_record_set = DataRecordSet([scan_dr], [record_op_stats])
        execution_data[scan_op_set_id][source_id] = [scan_record_set]

    # create execution data entries for first convert operator
    for model_idx in range(3):
        for source_idx in range(10):
            drs, record_op_stats_lst = [], []
            for one_to_many_idx in range(2):
                abs_idx = model_idx * 20 + source_idx * 2 + one_to_many_idx
                convert_dr = convert1_drs[abs_idx]
                source_id = convert_dr._source_id
                record_op_stats = RecordOpStats(
                    record_id=convert_dr._id,
                    record_parent_id=convert_dr._parent_id,
                    record_source_id=convert_dr._source_id,
                    op_id=f"convert1-phys-{str(models[model_idx])}",
                    op_name="LLMConvertBonded",
                    time_per_record=1.0,
                    cost_per_record=1.0,
                    logical_op_id=f"convert1-logical",
                    record_state=convert_dr._asDict(),
                    passed_filter=None,
                    generated_fields=["foo", "bar"],
                )
                drs.append(convert_dr)
                record_op_stats_lst.append(record_op_stats)
            convert_record_set = DataRecordSet(drs, record_op_stats_lst)
            if source_id not in execution_data[convert1_op_set_id]:
                execution_data[convert1_op_set_id][source_id] = [convert_record_set]
            else:
                execution_data[convert1_op_set_id][source_id].append(convert_record_set)

    # create execution data entries for first filter operator
    for model_idx in range(3):
        for source_idx in range(10):
            for one_to_many_idx in range(2):
                abs_idx = model_idx * 20 + source_idx * 2 + one_to_many_idx
                filter_dr = filter1_drs[abs_idx]
                source_id = filter_dr._source_id
                model = models[model_idx]

                # GPT-4 filters final 6 records it sees
                passed_filter = True
                if model_idx == 0 and source_idx > 6:
                    passed_filter = False

                # GPT-3.5 filters all records with one_to_many_idx == 1
                elif model_idx == 1 and one_to_many_idx == 1:
                    passed_filter = False

                # MIXTRAL passes all records

                record_op_stats = RecordOpStats(
                    record_id=filter_dr._id,
                    record_parent_id=filter_dr._parent_id,
                    record_source_id=filter_dr._source_id,
                    op_id=f"filter1-phys-{str(model)}",
                    op_name="LLMFilter",
                    time_per_record=1.0,
                    cost_per_record=1.0,
                    logical_op_id=f"filter1-logical",
                    record_state=filter_dr._asDict(),
                    passed_filter=passed_filter,
                    generated_fields=None,
                )
                filter_record_set = DataRecordSet([filter_dr], [record_op_stats])
                if source_id not in execution_data[filter1_op_set_id]:
                    execution_data[filter1_op_set_id][source_id] = [filter_record_set]
                else:
                    execution_data[filter1_op_set_id][source_id].append(filter_record_set)

    # create execution data entries for second filter operator
    for model_idx in range(3):
        for source_idx in range(7):
            for one_to_many_idx in range(2):
                abs_idx = model_idx * 14 + source_idx * 2 + one_to_many_idx
                filter_dr = filter2_drs[abs_idx]
                source_id = filter_dr._source_id
                model = models[model_idx]

                # TODO: this makes # of records seen by convert2 more complicated
                # GPT-4 filters out final 4 records it sees
                passed_filter = True
                if model_idx == 0 and source_idx > 4:
                    passed_filter = False

                # GPT-3.5 filters all records with one_to_many_idx == 1
                elif model_idx == 1 and one_to_many_idx == 1:
                    passed_filter = False

                # MIXTRAL passes all records

                # filter out records with abs_idx >= 30
                record_op_stats = RecordOpStats(
                    record_id=filter_dr._id,
                    record_parent_id=filter_dr._parent_id,
                    record_source_id=filter_dr._source_id,
                    op_id=f"filter2-phys-{str(model)}",
                    op_name="LLMFilter",
                    time_per_record=1.0,
                    cost_per_record=1.0,
                    logical_op_id=f"filter2-logical",
                    record_state=filter_dr._asDict(),
                    passed_filter=passed_filter,
                    generated_fields=None,
                )
                filter_record_set = DataRecordSet([filter_dr], [record_op_stats])
                if source_id not in execution_data[filter2_op_set_id]:
                    execution_data[filter2_op_set_id][source_id] = [filter_record_set]
                else:
                    execution_data[filter2_op_set_id][source_id].append(filter_record_set)

    # create execution data entries for second convert operator
    for model_idx in range(3):
        for source_idx in range(5):
            for one_to_many_idx in range(2):
                abs_idx = model_idx * 10 + source_idx * 2 + one_to_many_idx
                convert_dr = convert2_drs[abs_idx]
                source_id = convert_dr._source_id
                record_op_stats = RecordOpStats(
                    record_id=convert_dr._id,
                    record_parent_id=convert_dr._parent_id,
                    record_source_id=convert_dr._source_id,
                    op_id=f"convert1-phys-{str(models[model_idx])}",
                    op_name="LLMConvertBonded",
                    time_per_record=1.0,
                    cost_per_record=1.0,
                    logical_op_id=f"convert2-logical",
                    record_state=convert_dr._asDict(),
                    passed_filter=None,
                    generated_fields=["baz"],
                )
                convert_record_set = DataRecordSet([convert_dr], [record_op_stats])
                if source_id not in execution_data[convert2_op_set_id]:
                    execution_data[convert2_op_set_id][source_id] = [convert_record_set]
                else:
                    execution_data[convert2_op_set_id][source_id].append(convert_record_set)

    return execution_data
