import pytest

from palimpzest.constants import Model

# with open(".env") as f:
#     for line in f:
#         key, value = line.strip().split("=")
#         os.environ[key] = value

pytest_plugins = [
    "fixtures.cost_est_data",
    "fixtures.datasets",
    "fixtures.expected_cost_est_results",
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
    scan_only_plan,
    non_llm_filter_plan,
    llm_filter_plan,
    bonded_llm_convert_plan,
    code_synth_convert_plan,
    token_reduction_convert_plan,
    image_convert_plan,
    one_to_many_convert_plan,
    simple_plan_factory,
):
    physical_plan_id = request.param
    physical_plan_id_to_physical_plan = {
        "scan-only": scan_only_plan,
        "non-llm-filter": non_llm_filter_plan,
        "llm-filter": llm_filter_plan,
        "bonded-llm-convert": bonded_llm_convert_plan,
        "code-synth-convert": code_synth_convert_plan,
        "token-reduction-convert": token_reduction_convert_plan,
        "image-convert": image_convert_plan,
        "one-to-many-convert": one_to_many_convert_plan,
        "cost-est-simple-plan-gpt4-gpt4": simple_plan_factory(convert_model=Model.GPT_4, filter_model=Model.GPT_4),
        "cost-est-simple-plan-gpt4-gpt35": simple_plan_factory(convert_model=Model.GPT_4, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-gpt4-mixtral": simple_plan_factory(convert_model=Model.GPT_4, filter_model=Model.MIXTRAL),
        "cost-est-simple-plan-gpt35-gpt4": simple_plan_factory(convert_model=Model.GPT_3_5, filter_model=Model.GPT_4),
        "cost-est-simple-plan-gpt35-gpt35": simple_plan_factory(convert_model=Model.GPT_3_5, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-gpt35-mixtral": simple_plan_factory(convert_model=Model.GPT_3_5, filter_model=Model.MIXTRAL),
        "cost-est-simple-plan-mixtral-gpt4": simple_plan_factory(convert_model=Model.MIXTRAL, filter_model=Model.GPT_4),
        "cost-est-simple-plan-mixtral-gpt35": simple_plan_factory(convert_model=Model.MIXTRAL, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-mixtral-mixtral": simple_plan_factory(convert_model=Model.MIXTRAL, filter_model=Model.MIXTRAL),
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


@pytest.fixture
def expected_cost_est_results(
    request,
    simple_plan_expected_results_factory,
):
    cost_est_results_id = request.param
    cost_est_results_id_to_cost_est_results = {
        "cost-est-simple-plan-gpt4-gpt4": simple_plan_expected_results_factory(convert_model=Model.GPT_4, filter_model=Model.GPT_4),
        "cost-est-simple-plan-gpt4-gpt35": simple_plan_expected_results_factory(convert_model=Model.GPT_4, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-gpt4-mixtral": simple_plan_expected_results_factory(convert_model=Model.GPT_4, filter_model=Model.MIXTRAL),
        "cost-est-simple-plan-gpt35-gpt4": simple_plan_expected_results_factory(convert_model=Model.GPT_3_5, filter_model=Model.GPT_4),
        "cost-est-simple-plan-gpt35-gpt35": simple_plan_expected_results_factory(convert_model=Model.GPT_3_5, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-gpt35-mixtral": simple_plan_expected_results_factory(convert_model=Model.GPT_3_5, filter_model=Model.MIXTRAL),
        "cost-est-simple-plan-mixtral-gpt4": simple_plan_expected_results_factory(convert_model=Model.MIXTRAL, filter_model=Model.GPT_4),
        "cost-est-simple-plan-mixtral-gpt35": simple_plan_expected_results_factory(convert_model=Model.MIXTRAL, filter_model=Model.GPT_3_5),
        "cost-est-simple-plan-mixtral-mixtral": simple_plan_expected_results_factory(convert_model=Model.MIXTRAL, filter_model=Model.MIXTRAL),
    }

    return cost_est_results_id_to_cost_est_results[cost_est_results_id]
