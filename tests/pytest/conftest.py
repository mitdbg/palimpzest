import pytest

from palimpzest.constants import Model
from palimpzest.policy import MaxQuality, MaxQualityAtFixedCost, MinCost, MinCostAtFixedQuality

pytest_plugins = [
    "fixtures.champion_outputs",
    "fixtures.cost_est_data",
    "fixtures.datasets",
    "fixtures.execution_data",
    "fixtures.expected_cost_est_results",
    "fixtures.expected_physical_plans",
    "fixtures.expected_qualities",
    "fixtures.expected_records",
    "fixtures.physical_plans",
    "fixtures.operator_to_stats",
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
def dataset(request, enron_eval_tiny, real_estate_eval_tiny):
    dataset_id = request.param
    dataset_id_to_dataset = {
        "enron-eval-tiny": enron_eval_tiny,
        "real-estate-eval-tiny": real_estate_eval_tiny,
    }
    return dataset_id_to_dataset[dataset_id]


@pytest.fixture
def workload(
    request,
    enron_workload,
    real_estate_workload,
    three_converts_workload,
    one_filter_one_convert_workload,
    two_converts_two_filters_workload,
):
    workload_id = request.param
    workload_id_to_workload = {
        "enron-workload": enron_workload,
        "real-estate-workload": real_estate_workload,
        "three-converts": three_converts_workload,
        "one-filter-one-convert": one_filter_one_convert_workload,
        "two-converts-two-filters": two_converts_two_filters_workload,
    }
    return workload_id_to_workload[workload_id]


@pytest.fixture
def policy(request):
    policy_id = request.param
    policy_id_to_policy = {
        "mincost": MinCost(),
        "maxquality": MaxQuality(),
        "mincost@quality=0.8": MinCostAtFixedQuality(0.8),
        "maxquality@cost=1.0": MaxQualityAtFixedCost(1.0),
    }
    return policy_id_to_policy[policy_id]


@pytest.fixture
def physical_plan(
    request,
    scan_only_plan,
    non_llm_filter_plan,
    llm_filter_plan,
    bonded_llm_convert_plan,
    code_synth_convert_plan,
    rag_convert_plan,
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
        "rag-convert": rag_convert_plan,
        "image-convert": image_convert_plan,
        "one-to-many-convert": one_to_many_convert_plan,
        "cost-est-simple-plan-gpt4-gpt4": simple_plan_factory(
            convert_model=Model.GPT_4o, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-gpt4-gpt4m": simple_plan_factory(
            convert_model=Model.GPT_4o, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-gpt4-mixtral": simple_plan_factory(
            convert_model=Model.GPT_4o, filter_model=Model.MIXTRAL,
        ),
        "cost-est-simple-plan-gpt4m-gpt4": simple_plan_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-gpt4m-gpt4m": simple_plan_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-gpt4m-mixtral": simple_plan_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.MIXTRAL,
        ),
        "cost-est-simple-plan-mixtral-gpt4": simple_plan_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-mixtral-gpt4m": simple_plan_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-mixtral-mixtral": simple_plan_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.MIXTRAL,
        ),
    }
    return physical_plan_id_to_physical_plan[physical_plan_id]


@pytest.fixture
def sentinel_plan(
    request,
    scan_convert_filter_sentinel_plan,
    scan_multi_convert_multi_filter_sentinel_plan
):
    sentinel_plan_id = request.param
    sentinel_plan_id_to_sentinel_plan = {
        "scf": scan_convert_filter_sentinel_plan,
        "scffc": scan_multi_convert_multi_filter_sentinel_plan,
    }
    return sentinel_plan_id_to_sentinel_plan[sentinel_plan_id]

@pytest.fixture
def execution_data(
    request,
    scan_convert_filter_execution_data,
    scan_convert_filter_varied_execution_data,
    scan_multi_convert_multi_filter_execution_data,
):
    execution_data_id = request.param
    execution_data_id_to_execution_data = {
        "scf": scan_convert_filter_execution_data,
        "scf-varied": scan_convert_filter_varied_execution_data,
        "scffc": scan_multi_convert_multi_filter_execution_data,
    }
    return execution_data_id_to_execution_data[execution_data_id]

@pytest.fixture
def expected_records(
    request,
    enron_all_expected_records,
    enron_filter_expected_records,
    real_estate_all_expected_records,
    real_estate_one_to_many_expected_records,
    scan_convert_filter_expected_outputs,
    scan_convert_filter_empty_expected_outputs,
    scan_convert_filter_varied_expected_outputs,
    scan_multi_convert_multi_filter_expected_outputs,
):
    records_id = request.param
    records_id_to_expected_records = {
        "enron-all-records": enron_all_expected_records,
        "enron-filtered-records": enron_filter_expected_records,
        "real-estate-all-records": real_estate_all_expected_records,
        "real-estate-one-to-many-records": real_estate_one_to_many_expected_records,
        "scf": scan_convert_filter_expected_outputs,
        "empty": scan_convert_filter_empty_expected_outputs,
        "scf-varied": scan_convert_filter_varied_expected_outputs,
        "scffc": scan_multi_convert_multi_filter_expected_outputs,
    }
    return records_id_to_expected_records[records_id]


@pytest.fixture
def champion_outputs(
    request,
    scan_convert_filter_champion_outputs,
    scan_convert_filter_empty_champion_outputs,
    scan_convert_filter_varied_champion_outputs,
    scan_multi_convert_multi_filter_champion_outputs
):
    champion_outputs_id = request.param
    champion_outputs_id_to_champion_outputs = {
        "scf": scan_convert_filter_champion_outputs,
        "empty": scan_convert_filter_empty_champion_outputs,
        "scf-varied": scan_convert_filter_varied_champion_outputs,
        "scffc": scan_multi_convert_multi_filter_champion_outputs,
    }
    return champion_outputs_id_to_champion_outputs[champion_outputs_id]


@pytest.fixture
def expected_qualities(
    request,
    scan_convert_filter_qualities,
    scan_convert_filter_empty_qualities,
    scan_convert_filter_varied_qualities,
    scan_convert_filter_varied_override_qualities,
    scan_multi_convert_multi_filter_qualities,
):
    expected_qualities_id = request.param
    expected_qualities_id_to_expected_qualities = {
        "scf": scan_convert_filter_qualities,
        "empty": scan_convert_filter_empty_qualities,
        "scf-varied": scan_convert_filter_varied_qualities,
        "scf-varied-override": scan_convert_filter_varied_override_qualities,
        "scffc": scan_multi_convert_multi_filter_qualities,
    }
    return expected_qualities_id_to_expected_qualities[expected_qualities_id]


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
        "cost-est-simple-plan-gpt4-gpt4": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-gpt4-gpt4m": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-gpt4-mixtral": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o, filter_model=Model.MIXTRAL,
        ),
        "cost-est-simple-plan-gpt4m-gpt4": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-gpt4m-gpt4m": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-gpt4m-mixtral": simple_plan_expected_results_factory(
            convert_model=Model.GPT_4o_MINI, filter_model=Model.MIXTRAL,
        ),
        "cost-est-simple-plan-mixtral-gpt4": simple_plan_expected_results_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.GPT_4o,
        ),
        "cost-est-simple-plan-mixtral-gpt4m": simple_plan_expected_results_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.GPT_4o_MINI,
        ),
        "cost-est-simple-plan-mixtral-mixtral": simple_plan_expected_results_factory(
            convert_model=Model.MIXTRAL, filter_model=Model.MIXTRAL,
        ),
    }

    return cost_est_results_id_to_cost_est_results[cost_est_results_id]


@pytest.fixture
def operator_to_stats(
    request,
    three_converts_min_cost_operator_to_stats,
    three_converts_max_quality_operator_to_stats,
    three_converts_min_cost_at_fixed_quality_operator_to_stats,
    three_converts_max_quality_at_fixed_cost_operator_to_stats,
    one_filter_one_convert_min_cost_operator_to_stats,
    two_converts_two_filters_min_cost_operator_to_stats,
    two_converts_two_filters_max_quality_operator_to_stats,
    two_converts_two_filters_min_cost_at_fixed_quality_operator_to_stats,
    two_converts_two_filters_max_quality_at_fixed_cost_operator_to_stats,
):
    operator_to_stats_id = request.param
    operator_to_stats_id_to_operator_to_stats = {
        "3c-mincost": three_converts_min_cost_operator_to_stats,
        "3c-maxquality": three_converts_max_quality_operator_to_stats,
        "3c-mincost@quality=0.8": three_converts_min_cost_at_fixed_quality_operator_to_stats,
        "3c-maxquality@cost=1.0": three_converts_max_quality_at_fixed_cost_operator_to_stats,
        "1f-1c-mincost": one_filter_one_convert_min_cost_operator_to_stats,
        "2c-2f-mincost": two_converts_two_filters_min_cost_operator_to_stats,
        "2c-2f-maxquality": two_converts_two_filters_max_quality_operator_to_stats,
        "2c-2f-mincost@quality=0.8": two_converts_two_filters_min_cost_at_fixed_quality_operator_to_stats,
        "2c-2f-maxquality@cost=1.0": two_converts_two_filters_max_quality_at_fixed_cost_operator_to_stats,
    }

    return operator_to_stats_id_to_operator_to_stats[operator_to_stats_id]


@pytest.fixture
def expected_plan(
    request,
    three_converts_min_cost_expected_plan,
    three_converts_max_quality_expected_plan,
    three_converts_min_cost_at_fixed_quality_expected_plan,
    three_converts_max_quality_at_fixed_cost_expected_plan,
    one_filter_one_convert_min_cost_expected_plan,
    two_converts_two_filters_min_cost_expected_plan,
    two_converts_two_filters_max_quality_expected_plan,
    two_converts_two_filters_min_cost_at_fixed_quality_expected_plan,
    two_converts_two_filters_max_quality_at_fixed_cost_expected_plan,
):
    expected_plan_id = request.param
    expected_plan_id_to_expected_plan = {
        "3c-mincost": three_converts_min_cost_expected_plan,
        "3c-maxquality": three_converts_max_quality_expected_plan,
        "3c-mincost@quality=0.8": three_converts_min_cost_at_fixed_quality_expected_plan,
        "3c-maxquality@cost=1.0": three_converts_max_quality_at_fixed_cost_expected_plan,
        "1f-1c-mincost": one_filter_one_convert_min_cost_expected_plan,
        "2c-2f-mincost": two_converts_two_filters_min_cost_expected_plan,
        "2c-2f-maxquality": two_converts_two_filters_max_quality_expected_plan,
        "2c-2f-mincost@quality=0.8": two_converts_two_filters_min_cost_at_fixed_quality_expected_plan,
        "2c-2f-maxquality@cost=1.0": two_converts_two_filters_max_quality_at_fixed_cost_expected_plan,
    }

    return expected_plan_id_to_expected_plan[expected_plan_id]