import pytest

from palimpzest.constants import Model
from palimpzest.query.operators.convert import ConvertOp
from palimpzest.query.operators.datasource import MarshalAndScanDataOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.utils.model_helpers import get_models


@pytest.fixture
def simple_plan_expected_operator_estimates(simple_plan_scan_data, simple_plan_convert_data, simple_plan_filter_data):
    # get op_id for each operator
    scan_op_id = simple_plan_scan_data["op_id"]
    convert_op_id = simple_plan_convert_data["op_id"]
    filter_op_id = simple_plan_filter_data["op_id"]

    # get set of model names
    all_model_names = [m.value for m in get_models(include_vision=True)] + [None]

    # initialize expected operator estimates
    expected_operator_estimates = {
        scan_op_id: {
            "time_per_record": None,
        },
        convert_op_id: {
            model_name: {
                "time_per_record": None,
                "cost_per_record": None,
                "total_input_tokens": None,
                "total_output_tokens": None,
                "selectivity": None,
                "quality": None,
            }
            for model_name in all_model_names
        },
        filter_op_id: {
            model_name: {
                "time_per_record": None,
                "cost_per_record": None,
                "total_input_tokens": None,
                "total_output_tokens": None,
                "selectivity": None,
                "quality": None,
            }
            for model_name in all_model_names
        },
    }

    # fill-in scan operator estimates
    scan_time_per_records = simple_plan_scan_data["time_per_records"]
    expected_operator_estimates[scan_op_id]["time_per_record"] = sum(scan_time_per_records) / len(scan_time_per_records)

    # fill-in convert operator estimates
    convert_time_per_records = simple_plan_convert_data["time_per_records"]
    convert_cost_per_records = simple_plan_convert_data["cost_per_records"]
    convert_total_input_tokens = simple_plan_convert_data["total_input_tokens"]
    convert_total_output_tokens = simple_plan_convert_data["total_output_tokens"]
    for model_name in all_model_names:
        model_start_idx, model_end_idx, expected_quality = None, None, None
        if model_name == Model.GPT_4o.value:
            model_start_idx = 0
            model_end_idx = 2
            expected_quality = 1.0
        elif model_name == Model.GPT_4o_MINI.value:
            model_start_idx = 2
            model_end_idx = 4
            expected_quality = 2.0 / 4.0  # correct answers / total keys
        elif model_name == Model.MIXTRAL.value:
            model_start_idx = 4
            model_end_idx = 6
            expected_quality = 1.0 / 4.0  # correct answers / total keys
        else:
            model_start_idx = 0
            model_end_idx = 6
            expected_quality = 7.0 / 12.0  # correct answers / total keys

        num_samples = model_end_idx - model_start_idx
        expected_operator_estimates[convert_op_id][model_name]["time_per_record"] = (
            sum(convert_time_per_records[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[convert_op_id][model_name]["cost_per_record"] = (
            sum(convert_cost_per_records[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[convert_op_id][model_name]["total_input_tokens"] = (
            sum(convert_total_input_tokens[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[convert_op_id][model_name]["total_output_tokens"] = (
            sum(convert_total_output_tokens[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[convert_op_id][model_name]["selectivity"] = 1.0
        expected_operator_estimates[convert_op_id][model_name]["quality"] = expected_quality

    # fill-in filter operator estimates
    filter_time_per_records = simple_plan_filter_data["time_per_records"]
    filter_cost_per_records = simple_plan_filter_data["cost_per_records"]
    filter_total_input_tokens = simple_plan_filter_data["total_input_tokens"]
    filter_total_output_tokens = simple_plan_filter_data["total_output_tokens"]
    for model_name in all_model_names:
        model_start_idx, model_end_idx, expected_selectivity, expected_quality = None, None, None, None
        if model_name == Model.GPT_4o.value:
            model_start_idx = 0
            model_end_idx = 2
            expected_selectivity = 0.5
            expected_quality = 1.0
        elif model_name == Model.GPT_4o_MINI.value:
            model_start_idx = 2
            model_end_idx = 4
            expected_selectivity = 0.5
            expected_quality = 0.0  # avg. accuracy
        elif model_name == Model.MIXTRAL.value:
            model_start_idx = 4
            model_end_idx = 6
            expected_selectivity = 1.0
            expected_quality = 0.5  # avg. accuracy
        else:
            model_start_idx = 0
            model_end_idx = 6
            expected_selectivity = 4.0 / 6.0
            expected_quality = 3.0 / 6.0  # avg. accuracy

        num_samples = model_end_idx - model_start_idx
        expected_operator_estimates[filter_op_id][model_name]["time_per_record"] = (
            sum(filter_time_per_records[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[filter_op_id][model_name]["cost_per_record"] = (
            sum(filter_cost_per_records[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[filter_op_id][model_name]["total_input_tokens"] = (
            sum(filter_total_input_tokens[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[filter_op_id][model_name]["total_output_tokens"] = (
            sum(filter_total_output_tokens[model_start_idx:model_end_idx]) / num_samples
        )
        expected_operator_estimates[filter_op_id][model_name]["selectivity"] = expected_selectivity
        expected_operator_estimates[filter_op_id][model_name]["quality"] = expected_quality

    return expected_operator_estimates


@pytest.fixture
def simple_plan_expected_results_factory(simple_plan_expected_operator_estimates):
    # the factory must return a function which is parameterized by convert_model and filter_model
    def expected_results_generator(convert_model, filter_model):
        # the generator returns a function which is parameterized by the input_cardinality,
        # because the input_cardinality is provided at test time by the test class
        def expected_results_fn(physical_op, input_cardinality):
            expected_op_time, expected_op_cost, expected_op_quality, output_cardinality = None, None, None, None

            # compute expected cost, time, and quality for different operations
            if isinstance(physical_op, MarshalAndScanDataOp):
                scan_time_per_record = simple_plan_expected_operator_estimates["scan123"]["time_per_record"]
                expected_op_time = scan_time_per_record * input_cardinality
                expected_op_cost = 0.0
                expected_op_quality = 1.0
                scan_selectivity = 1.0
                output_cardinality = scan_selectivity * input_cardinality

            # compute expected cost, time, and quality for convert operation
            elif isinstance(physical_op, ConvertOp):
                convert_time_per_record = simple_plan_expected_operator_estimates["convert123"][convert_model][
                    "time_per_record"
                ]
                expected_op_time = convert_time_per_record * input_cardinality

                convert_cost_per_record = simple_plan_expected_operator_estimates["convert123"][convert_model][
                    "cost_per_record"
                ]
                expected_op_cost = convert_cost_per_record * input_cardinality

                expected_op_quality = simple_plan_expected_operator_estimates["convert123"][convert_model]["quality"]
                convert_selectivity = simple_plan_expected_operator_estimates["convert123"][convert_model][
                    "selectivity"
                ]
                output_cardinality = convert_selectivity * input_cardinality

            # compute expected cost, time, and quality for filter operation
            elif isinstance(physical_op, FilterOp):
                filter_time_per_record = simple_plan_expected_operator_estimates["filter123"][filter_model][
                    "time_per_record"
                ]
                expected_op_time = filter_time_per_record * input_cardinality

                filter_cost_per_record = simple_plan_expected_operator_estimates["filter123"][filter_model][
                    "cost_per_record"
                ]
                expected_op_cost = filter_cost_per_record * input_cardinality

                expected_op_quality = simple_plan_expected_operator_estimates["filter123"][filter_model]["quality"]
                filter_selectivity = simple_plan_expected_operator_estimates["filter123"][filter_model]["selectivity"]
                output_cardinality = filter_selectivity * input_cardinality

            return expected_op_cost, expected_op_time, expected_op_quality, output_cardinality

        return expected_results_fn

    return expected_results_generator
