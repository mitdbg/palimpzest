import pytest
from palimpzest.constants import Model
from palimpzest.core.data.dataclasses import RecordOpStats


@pytest.fixture
def sample_op_data_factory():
    # this fixture returns a function which generates sample execution data as specified by the fcn. parameters
    def sample_op_data_generator(
            op_id,
            logical_op_id,
            op_name,
            source_op_id,
            plan_ids,
            record_ids,
            record_parent_ids,
            record_source_ids,
            time_per_records,
            cost_per_records,
            total_input_tokens=None,
            total_output_tokens=None,
            model_names=None,
            answers=None
        ):
        sample_op_data = [
            RecordOpStats(
                record_id=record_ids[idx],
                record_parent_id=record_parent_ids[idx] if record_parent_ids is not None else None,
                record_source_id=record_source_ids[idx],
                record_state={},
                op_id=op_id,
                logical_op_id=logical_op_id,
                op_name=op_name,
                time_per_record=time_per_records[idx],
                cost_per_record=cost_per_records[idx],
                total_input_tokens=total_input_tokens[idx] if total_input_tokens is not None else 0.0,
                total_output_tokens=total_output_tokens[idx] if total_output_tokens is not None else 0.0,
                source_op_id=source_op_id,
                plan_id=plan_ids[idx],
                model_name=model_names[idx] if model_names is not None else None,
                answer=answers[idx] if answers is not None else None,
                passed_operator=answers[idx] if "filter" in op_name.lower() else None,
            )
            for idx in range(len(time_per_records))
        ]
        return sample_op_data

    return sample_op_data_generator


@pytest.fixture
def simple_plan_scan_data():
    # we simulate scanning two records with three different plans
    return {
        "op_id": "scan123",
        "logical_op_id": "BaseScan",
        "op_name": "MarshalAndScanDataOp",
        "source_op_id": None,
        "plan_ids": ["plan1"] * 2 + ["plan2"] * 2 + ["plan3"] * 2,
        "record_ids": ["scan1", "scan2"] * 3,
        "record_parent_ids": None,
        "record_source_ids": ["source1", "source2"] * 3,
        "time_per_records": [1, 1, 2, 3, 5, 8],
        "cost_per_records": [0, 0, 0, 0, 0, 0],
        "total_input_tokens": None,
        "total_output_tokens": None,
        "model_names": None,
        "answers": None,
    }


@pytest.fixture
def simple_plan_convert_data(simple_plan_scan_data):
    # we simulate converting the records output by the simple plan's scan operation
    scan_record_ids = simple_plan_scan_data["record_ids"]
    return {
        "op_id": "convert123",
        "logical_op_id": "ConvertScan",
        "op_name": "LLMConvertBonded",
        "source_op_id": "scan123",
        "plan_ids": simple_plan_scan_data["plan_ids"],
        "record_ids": [id.replace("scan", "convert") for id in scan_record_ids],
        "record_parent_ids": scan_record_ids,
        "record_source_ids": ["source1", "source2"] * 3,
        "time_per_records": [1, 2, 4, 8, 16, 32],
        "cost_per_records": [2, 4, 6, 8, 10, 12],
        "total_input_tokens": [200, 400, 600, 800, 100, 1200],
        "total_output_tokens": [20, 40, 60, 80, 100, 120],
        "model_names": [Model.GPT_4o.value] * 2 + [Model.GPT_4o_MINI.value] * 2 + [Model.MIXTRAL.value] * 2,
        "answers": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        + [{"a": 1, "b": 1}, {"a": 3, "b": 3}]
        + [{"a": 1, "b": 0}, {"a": 0, "b": 0}],
    }


@pytest.fixture
def simple_plan_filter_data(simple_plan_convert_data):
    # we simulate filtering the records output by the simple plan's convert operation
    convert_record_ids = simple_plan_convert_data["record_ids"]
    return {
        "op_id": "filter123",
        "logical_op_id": "FilteredScan",
        "op_name": "LLMFilter",
        "source_op_id": "convert123",
        "plan_ids": simple_plan_convert_data["plan_ids"],
        "record_ids": [id.replace("convert", "filter") for id in convert_record_ids],
        "record_parent_ids": convert_record_ids,
        "record_source_ids": ["source1", "source2"] * 3,
        "time_per_records": [1, 3, 5, 7, 9, 11],
        "cost_per_records": [1, 2, 1, 2, 1, 2],
        "total_input_tokens": [100, 200, 100, 200, 100, 200],
        "total_output_tokens": [10, 20, 10, 20, 10, 20],
        "model_names": [Model.GPT_4o.value] * 2 + [Model.GPT_4o_MINI.value] * 2 + [Model.MIXTRAL.value] * 2,
        "answers": [True, False] + [False, True] + [True, True],
    }


@pytest.fixture
def simple_plan_sample_execution_data(
    simple_plan_scan_data,
    simple_plan_convert_data,
    simple_plan_filter_data,
    sample_op_data_factory,
):
    # combine sample op data for all operators
    all_op_data_params = [
        simple_plan_scan_data,
        simple_plan_convert_data,
        simple_plan_filter_data,
    ]

    # generate sample op data for all operators
    all_sample_execution_data = []
    for op_data_params in all_op_data_params:
        sample_op_data = sample_op_data_factory(**op_data_params)
        all_sample_execution_data.extend(sample_op_data)

    return all_sample_execution_data
