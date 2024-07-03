import pytest

# with open(".env") as f:
#     for line in f:
#         key, value = line.strip().split("=")
#         os.environ[key] = value

pytest_plugins = [
    "fixtures.datasets",
    "fixtures.expected_records",
    "fixtures.physical_plans",
    "fixtures.schemas",
    "fixtures.side_effects",
    "fixtures.workloads",
]

# NOTE: these fixtures may grow to have long lists of arguments;
#       the benefit of using fixtures here (which requires us to specify them
#       as arguments) is that pytest will compute each fixture value once
#       and cache the result. Thus, we minimize recomputation and don't
#       need to, for example, re-register datasets for each individual test.

@pytest.fixture
def dataset(request, enron_eval_tiny, real_estate_eval_tiny, biofabric_tiny):
    dataset_id = request.param
    dataset_id_to_dataset = {
        "enron-eval-tiny": enron_eval_tiny,
        "real-estate-eval-tiny": real_estate_eval_tiny,
        "biofabric-tiny": biofabric_tiny,
    }
    return dataset_id_to_dataset[dataset_id]


@pytest.fixture
def workload(
    request,
    enron_workload,
    real_estate_workload,
    biofabric_workload,
):
    workload_id = request.param
    workload_id_to_workload = {
        "enron-workload": enron_workload,
        "real-estate-workload": real_estate_workload,
        "biofabric-workload": biofabric_workload,
    }
    return workload_id_to_workload[workload_id]


@pytest.fixture
def physical_plan(
    request,
    enron_scan_only_plan,
    enron_non_llm_filter_plan,
    enron_llm_filter_plan,
    enron_bonded_llm_convert_plan,
    enron_code_synth_convert_plan,
    enron_token_reduction_convert_plan,
    real_estate_image_convert_plan,
    real_estate_one_to_many_convert_plan,
):
    physical_plan_id = request.param
    physical_plan_id_to_physical_plan = {
        "enron-scan-only": enron_scan_only_plan,
        "enron-non-llm-filter": enron_non_llm_filter_plan,
        "enron-llm-filter": enron_llm_filter_plan,
        "enron-bonded-llm-convert": enron_bonded_llm_convert_plan,
        "enron-code-synth-convert": enron_code_synth_convert_plan,
        "enron-token-reduction-convert": enron_token_reduction_convert_plan,
        "real-estate-image-convert": real_estate_image_convert_plan,
        "real-estate-one-to-many-convert": real_estate_one_to_many_convert_plan,
    }
    return physical_plan_id_to_physical_plan[physical_plan_id]


@pytest.fixture
def expected_records(
    request,
    enron_all_expected_records,
    enron_filter_expected_records,
    real_estate_all_expected_records,
    real_estate_one_to_many_expected_records,
):
    records_id = request.param
    records_id_to_expected_records = {
        "enron-all-records": enron_all_expected_records,
        "enron-filtered-records": enron_filter_expected_records,
        "real-estate-all-records": real_estate_all_expected_records,
        "real-estate-one-to-many-records": real_estate_one_to_many_expected_records,
    }
    return records_id_to_expected_records[records_id]


@pytest.fixture
def side_effect(
    request,
    enron_filter,
    enron_convert,
    real_estate_convert,
    real_estate_one_to_many_convert,
):
    side_effect_id = request.param
    side_effect_id_to_side_effect = {
        None: None,
        "enron-filter": enron_filter,
        "enron-convert": enron_convert,
        "real-estate-convert": real_estate_convert,
        "real-estate-one-to-many-convert": real_estate_one_to_many_convert,
    }
    return side_effect_id_to_side_effect[side_effect_id]
