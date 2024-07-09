from palimpzest.constants import Model
from palimpzest.utils import getModels

import pytest

@pytest.fixture
def simple_plan_expected_operator_estimates(simple_plan_scan_data, simple_plan_convert_data, simple_plan_filter_data):
    # get op_id for each operator
    scan_op_id = simple_plan_scan_data["op_id"]
    convert_op_id = simple_plan_convert_data["op_id"]
    filter_op_id = simple_plan_filter_data["op_id"]

    # get set of model names
    all_model_names = [m.value for m in getModels(include_vision=True)] + [None]

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
        if model_name == Model.GPT_4.value:
            model_start_idx = 0
            model_end_idx = 2
            expected_quality = 1.0
        elif model_name == Model.GPT_3_5.value:
            model_start_idx = 2
            model_end_idx = 4
            expected_quality = ((1.0 / 2.0) + (1.0 / 2.0)) / 2.0  # avg. recall (per-key)
        elif model_name == Model.MIXTRAL.value:
            model_start_idx = 4
            model_end_idx = 6
            expected_quality = ((1.0 / 2.0) + (0.0 / 2.0)) / 2.0  # avg. recall (per-key)
        else:
            model_start_idx = 0
            model_end_idx = 6
            expected_quality = ((2.0 / 2.0) + (2.0 / 2.0) + (1.0 / 2.0) + (1.0 / 2.0) + (1.0 / 2.0) + (0.0 / 2.0)) / 6.0 # avg. recall (per-key)

        num_samples = model_end_idx - model_start_idx
        expected_operator_estimates[convert_op_id][model_name]["time_per_record"] = sum(convert_time_per_records[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[convert_op_id][model_name]["cost_per_record"] = sum(convert_cost_per_records[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[convert_op_id][model_name]["total_input_tokens"] = sum(convert_total_input_tokens[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[convert_op_id][model_name]["total_output_tokens"] = sum(convert_total_output_tokens[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[convert_op_id][model_name]["selectivity"] = 1.0
        expected_operator_estimates[convert_op_id][model_name]["quality"] = expected_quality

    # fill-in filter operator estimates
    filter_time_per_records = simple_plan_filter_data["time_per_records"]
    filter_cost_per_records = simple_plan_filter_data["cost_per_records"]
    filter_total_input_tokens = simple_plan_filter_data["total_input_tokens"]
    filter_total_output_tokens = simple_plan_filter_data["total_output_tokens"]
    for model_name in all_model_names:
        model_start_idx, model_end_idx, expected_selectivity, expected_quality = None, None, None, None
        if model_name == Model.GPT_4.value:
            model_start_idx = 0
            model_end_idx = 2
            expected_selectivity = 0.5
            expected_quality = 1.0
        elif model_name == Model.GPT_3_5.value:
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
        expected_operator_estimates[filter_op_id][model_name]["time_per_record"] = sum(filter_time_per_records[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[filter_op_id][model_name]["cost_per_record"] = sum(filter_cost_per_records[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[filter_op_id][model_name]["total_input_tokens"] = sum(filter_total_input_tokens[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[filter_op_id][model_name]["total_output_tokens"] = sum(filter_total_output_tokens[model_start_idx:model_end_idx])/num_samples
        expected_operator_estimates[filter_op_id][model_name]["selectivity"] = expected_selectivity
        expected_operator_estimates[filter_op_id][model_name]["quality"] = expected_quality

    return expected_operator_estimates


@pytest.fixture
def simple_plan_expected_results_factory(simple_plan_expected_operator_estimates):
    # the factory must return a function which is parameterized by convert_model and filter_model
    def expected_results_generator(convert_model, filter_model):
        # the generator returns a function which is parameterized by the input_cardinality,
        # because the input_cardinality is provided at test time by the test class
        def expected_results_fn(input_cardinality):
            # compute expected time, cost, and quality for scan operation
            scan_time_per_record = simple_plan_expected_operator_estimates["scan123"]["time_per_record"]
            expected_scan_time = scan_time_per_record * input_cardinality
            expected_scan_cost = 0.0
            expected_scan_quality = 1.0
            scan_selectivity = 1.0

            # compute expected time, cost, and quality for convert operation
            input_cardinality = scan_selectivity * input_cardinality
            convert_time_per_record = simple_plan_expected_operator_estimates["convert123"][convert_model]["time_per_record"]
            expected_convert_time = convert_time_per_record * input_cardinality

            convert_cost_per_record = simple_plan_expected_operator_estimates["convert123"][convert_model]["cost_per_record"]
            expected_convert_cost = convert_cost_per_record * input_cardinality

            expected_convert_quality = simple_plan_expected_operator_estimates["convert123"][convert_model]["quality"]
            convert_selectivity = simple_plan_expected_operator_estimates["convert123"][convert_model]["selectivity"]

            # compute expected time, cost, and quality for filter operation
            input_cardinality = convert_selectivity * input_cardinality
            filter_time_per_record = simple_plan_expected_operator_estimates["filter123"][filter_model]["time_per_record"]
            expected_filter_time = filter_time_per_record * input_cardinality

            filter_cost_per_record = simple_plan_expected_operator_estimates["filter123"][filter_model]["cost_per_record"]
            expected_filter_cost = filter_cost_per_record * input_cardinality

            expected_filter_quality = simple_plan_expected_operator_estimates["filter123"][filter_model]["quality"]

            # compute aggregate time, cost, and quality
            total_time = expected_scan_time + expected_convert_time + expected_filter_time
            total_cost = expected_scan_cost + expected_convert_cost + expected_filter_cost
            quality = expected_scan_quality * expected_convert_quality * expected_filter_quality

            return total_time, total_cost, quality
        
        return expected_results_fn

    return expected_results_generator
