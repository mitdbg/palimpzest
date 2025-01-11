import pytest
from palimpzest.constants import Model
from palimpzest.core.data.dataclasses import PlanCost
from palimpzest.core.elements.filters import Filter
from palimpzest.core.lib.schemas import TextFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.datasource import MarshalAndScanDataOp
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.logical import BaseScan, ConvertScan, FilteredScan
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.sets import Dataset


### THREE CONVERTS PHYSICAL PLANS ###
def get_three_converts_plan(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema, models, expected_cost, expected_time, expected_quality):
    # extract node id's from workload Datasets
    scan_node_id = three_converts_workload._source._source._source.universal_identifier()
    first_convert_node_id = three_converts_workload._source._source.universal_identifier()
    second_convert_node_id = three_converts_workload._source.universal_identifier()
    third_convert_node_id = three_converts_workload.universal_identifier()

    # create physical op for scan operator
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny, logical_op_id=scan_logical_op.get_logical_op_id())

    # create physical op for first convert operator
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_op = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=models[0], depends_on=list(depends_on), logical_op_id=first_convert_logical_op.get_logical_op_id())

    # get physical op id for second convert operators
    depends_on.update(first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id))
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=foobar_schema, depends_on=list(depends_on), target_cache_id=second_convert_node_id)
    second_convert_op = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=models[1], depends_on=list(depends_on), logical_op_id=second_convert_logical_op.get_logical_op_id())

    # get physical op id for third convert operators
    depends_on.update(second_convert_logical_op.output_schema.field_names(unique=True, id=second_convert_node_id))
    third_convert_logical_op = ConvertScan(input_schema=foobar_schema, output_schema=baz_schema, depends_on=list(depends_on), target_cache_id=third_convert_node_id)
    third_convert_op = LLMConvertBonded(output_schema=baz_schema, input_schema=foobar_schema, model=models[2], depends_on=list(depends_on), logical_op_id=third_convert_logical_op.get_logical_op_id())

    plan = PhysicalPlan(
        operators=[scan_op, first_convert_op, second_convert_op, third_convert_op],
        plan_cost=PlanCost(cost=expected_cost, time=expected_time, quality=expected_quality),
    )
    return plan

@pytest.fixture
def three_converts_min_cost_expected_plan(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = 0.3 * cardinality
    expected_time = 4.0 * cardinality
    expected_quality = 1.0

    return get_three_converts_plan(
        three_converts_workload=three_converts_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        baz_schema=baz_schema,
        models=[Model.LLAMA3, Model.GPT_4o_MINI, Model.LLAMA3],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def three_converts_max_quality_expected_plan(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = 3.0 * cardinality
    expected_time = 4.0 * cardinality
    expected_quality = 0.81

    return get_three_converts_plan(
        three_converts_workload=three_converts_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        baz_schema=baz_schema,
        models=[Model.GPT_4o, Model.GPT_4o_MINI, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def three_converts_min_cost_at_fixed_quality_expected_plan(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = 2.0 * cardinality
    expected_time = 4.0 * cardinality
    expected_quality = 0.81

    return get_three_converts_plan(
        three_converts_workload=three_converts_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        baz_schema=baz_schema,
        models=[Model.LLAMA3, Model.GPT_4o_MINI, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def three_converts_max_quality_at_fixed_cost_expected_plan(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (0.9 / cardinality) * cardinality
    expected_time = 4.0 * cardinality
    expected_quality = 0.72

    return get_three_converts_plan(
        three_converts_workload=three_converts_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        baz_schema=baz_schema,
        models=[Model.GPT_4o_MINI, Model.GPT_4o_MINI, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

### ONE FILTER ONE CONVERT PHYSICAL PLANS ###
def get_one_filter_one_convert_plan(one_filter_one_convert_workload, enron_eval_tiny, email_schema, models, expected_cost, expected_time, expected_quality):
    dataset_nodes = []
    node = one_filter_one_convert_workload.copy()
    while isinstance(node, Dataset):
        dataset_nodes.append(node)
        node = node._source
    dataset_nodes.append(node)
    dataset_nodes = list(reversed(dataset_nodes))

    # remove unnecessary convert because output schema from data source scan matches
    # input schema for the next operator
    if len(dataset_nodes) > 1 and dataset_nodes[0].schema == dataset_nodes[1].schema:
        dataset_nodes = [dataset_nodes[0]] + dataset_nodes[2:]
        if len(dataset_nodes) > 1:
            dataset_nodes[1]._source = dataset_nodes[0]

    # extract node id's from workload Datasets
    scan_node_id = dataset_nodes[0].universal_identifier()
    first_filter_node_id = dataset_nodes[1].universal_identifier()
    first_convert_node_id = dataset_nodes[2].universal_identifier()

    # create physical op for scan operator
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny, logical_op_id=scan_logical_op.get_logical_op_id())

    # get physical op id for first filter operator
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_filter_logical_op = FilteredScan(input_schema=TextFile, output_schema=TextFile, filter=Filter("filter1"), depends_on=list(depends_on), target_cache_id=first_filter_node_id)
    first_filter_op = LLMFilter(output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=models[0], depends_on=list(depends_on), logical_op_id=first_filter_logical_op.get_logical_op_id())

    # create physical op for first convert operator
    depends_on = depends_on.union(set(first_filter_logical_op.output_schema.field_names(unique=True, id=first_filter_node_id)))
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_op = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=models[1], depends_on=list(depends_on), logical_op_id=first_convert_logical_op.get_logical_op_id())

    plan = PhysicalPlan(
        operators=[scan_op, first_filter_op, first_convert_op],
        plan_cost=PlanCost(cost=expected_cost, time=expected_time, quality=expected_quality),
    )
    return plan

@pytest.fixture
def one_filter_one_convert_min_cost_expected_plan(one_filter_one_convert_workload, enron_eval_tiny, email_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (
        1.0 * 1.0 * cardinality
        + 1.0 * 0.5 * cardinality
    )
    expected_time = (
        2.0 * cardinality
        + 1.0 * 0.5 * cardinality
    )
    expected_quality = 1.0

    return get_one_filter_one_convert_plan(
        one_filter_one_convert_workload=one_filter_one_convert_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        models=[Model.LLAMA3, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

### TWO CONVERTS TWO FILTERS PHYSICAL PLANS ###
def get_two_converts_two_filters_plan(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, first_filter_str, models, expected_cost, expected_time, expected_quality):
    # extract node id's from workload Datasets
    scan_node_id = two_converts_two_filters_workload._source._source._source._source.universal_identifier()
    first_convert_node_id = two_converts_two_filters_workload._source._source._source.universal_identifier()
    second_convert_node_id = two_converts_two_filters_workload._source._source.universal_identifier()
    first_filter_node_id = two_converts_two_filters_workload._source.universal_identifier()
    second_filter_node_id = two_converts_two_filters_workload.universal_identifier()

    # create physical op for scan operator
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny, logical_op_id=scan_logical_op.get_logical_op_id())

    # create physical op for first convert operator
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_op = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=models[0], depends_on=list(depends_on), logical_op_id=first_convert_logical_op.get_logical_op_id())

    # get physical op id for second convert operators
    depends_on.update(first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id))
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=foobar_schema, depends_on=list(depends_on), target_cache_id=second_convert_node_id)
    second_convert_op = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=models[1], depends_on=list(depends_on), logical_op_id=second_convert_logical_op.get_logical_op_id())

    # get physical op id for first filter operator
    depends_on = [field for field in first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id) if "sender" in field]
    first_filter_logical_op = FilteredScan(input_schema=foobar_schema, output_schema=foobar_schema, filter=Filter("filter1"), depends_on=list(depends_on), target_cache_id=first_filter_node_id)
    first_filter_op = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter1"), model=models[2], depends_on=list(depends_on), logical_op_id=first_filter_logical_op.get_logical_op_id())

    # get physical op id for second filter operator
    depends_on = [field for field in first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id) if "subject" in field]
    second_filter_logical_op = FilteredScan(input_schema=foobar_schema, output_schema=foobar_schema, filter=Filter("filter2"), depends_on=list(depends_on), target_cache_id=second_filter_node_id)
    second_filter_op = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter2"), model=models[3], depends_on=list(depends_on), logical_op_id=second_filter_logical_op.get_logical_op_id())

    plan = PhysicalPlan(
        operators=(
            [scan_op, first_convert_op, first_filter_op, second_filter_op, second_convert_op]
            if first_filter_str == "filter1"
            else [scan_op, first_convert_op, second_filter_op, first_filter_op, second_convert_op]
        ),
        plan_cost=PlanCost(cost=expected_cost, time=expected_time, quality=expected_quality),
    )
    return plan

@pytest.fixture
def two_converts_two_filters_min_cost_expected_plan(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (
        0.1 * 1.0 * cardinality
        + 1.0 * 1.0 * cardinality
        + 1.0 * (1 / 3) * cardinality
        + 0.1 * (1 / 3) * 0.5 * cardinality
    )
    expected_time = (
        3.0 * cardinality
        + (1 / 3) * cardinality
        + (1 / 3) * 0.5 * cardinality
    )
    expected_quality = 1.0

    return get_two_converts_two_filters_plan(
        two_converts_two_filters_workload=two_converts_two_filters_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        first_filter_str="filter2",
        # models: [first convert, second convert, filter1, filter2]
        models=[Model.LLAMA3, Model.GPT_4o_MINI, Model.GPT_4o_MINI, Model.LLAMA3],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def two_converts_two_filters_max_quality_expected_plan(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (
        0.5 * cardinality
        + 1.0 * cardinality
        + 0.75 * 0.5 * cardinality
        + 1.0 * 0.5 * 0.75 * cardinality
    )
    expected_time = (
        3.0 * cardinality
        + 0.5 * cardinality
        + 0.5 * 0.75 * cardinality
    )
    expected_quality = 0.81

    return get_two_converts_two_filters_plan(
        two_converts_two_filters_workload=two_converts_two_filters_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        first_filter_str="filter2",
        # models: [first convert, second convert, filter1, filter2]
        models=[Model.LLAMA3, Model.GPT_4o_MINI, Model.LLAMA3, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def two_converts_two_filters_min_cost_at_fixed_quality_expected_plan(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (
        0.5 * cardinality
        + 1.0 * cardinality
        + 0.5 * 0.5 * cardinality
        + 0.3 * 0.5 * 0.75 * cardinality
    )
    expected_time = (
        3.0 * cardinality
        + 0.5 * cardinality
        + 0.5 * 0.75 * cardinality
    )
    expected_quality = 0.81

    return get_two_converts_two_filters_plan(
        two_converts_two_filters_workload=two_converts_two_filters_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        first_filter_str="filter1",
        # models: [first convert, second convert, filter1, filter2]
        models=[Model.GPT_4o_MINI, Model.LLAMA3, Model.GPT_4o_MINI, Model.LLAMA3],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )

@pytest.fixture
def two_converts_two_filters_max_quality_at_fixed_cost_expected_plan(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema):
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))
    expected_cost = (
        (0.3 / cardinality) * cardinality
        + (0.3 / cardinality) * cardinality
        + (0.3 / cardinality) * 0.5 * cardinality
        + (0.2 / cardinality) * 0.5 * cardinality
    )
    expected_time = (
        3.0 * cardinality
        + 0.5 * cardinality
        + 0.5 * cardinality
    )
    expected_quality = 0.81

    return get_two_converts_two_filters_plan(
        two_converts_two_filters_workload=two_converts_two_filters_workload,
        enron_eval_tiny=enron_eval_tiny,
        email_schema=email_schema,
        foobar_schema=foobar_schema,
        first_filter_str="filter2",
        # models: [first convert, second convert, filter1, filter2]
        models=[Model.GPT_4o_MINI, Model.LLAMA3, Model.GPT_4o, Model.GPT_4o],
        expected_cost=expected_cost,
        expected_time=expected_time,
        expected_quality=expected_quality,
    )
