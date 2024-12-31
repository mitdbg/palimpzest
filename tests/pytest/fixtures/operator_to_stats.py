import pytest

from palimpzest.constants import Model
from palimpzest.corelib.schemas import TextFile
from palimpzest.datamanager import DataDirectory
from palimpzest.elements.filters import Filter
from palimpzest.operators.convert import LLMConvertBonded
from palimpzest.operators.datasource import MarshalAndScanDataOp
from palimpzest.operators.filter import LLMFilter
from palimpzest.operators.logical import BaseScan, ConvertScan, FilteredScan
from palimpzest.sets import Dataset


### THREE CONVERTS OPERATOR-TO-STATS ###
def get_three_converts_logical_and_physical_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # extract node id's from workload Datasets
    scan_node_id = three_converts_workload._source._source._source.universal_identifier()
    first_convert_node_id = three_converts_workload._source._source.universal_identifier()
    second_convert_node_id = three_converts_workload._source.universal_identifier()
    third_convert_node_id = three_converts_workload.universal_identifier()

    # get physical and logical op id for scan operator
    scan_physical_op_id = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny).get_op_id()
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()

    # get physical op ids for first convert operators
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    first_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    first_convert_llama_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()

    # get physical op ids for second convert operators
    depends_on.update(first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id))
    second_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    second_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    second_convert_llama_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=foobar_schema, depends_on=list(depends_on), target_cache_id=second_convert_node_id)
    second_convert_logical_op_id = second_convert_logical_op.get_logical_op_id()

    # get physical op ids for third convert operators
    depends_on.update(second_convert_logical_op.output_schema.field_names(unique=True, id=second_convert_node_id))
    third_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=baz_schema, input_schema=foobar_schema, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    third_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=baz_schema, input_schema=foobar_schema, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    third_convert_llama_physical_op_id = LLMConvertBonded(output_schema=baz_schema, input_schema=foobar_schema, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    third_convert_logical_op = ConvertScan(input_schema=foobar_schema, output_schema=baz_schema, depends_on=list(depends_on), target_cache_id=third_convert_node_id)
    third_convert_logical_op_id = third_convert_logical_op.get_logical_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_physical_op_id": scan_physical_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_physical_op_id": first_convert_gpt4o_physical_op_id,
        "first_convert_gpt4o_mini_physical_op_id": first_convert_gpt4o_mini_physical_op_id,
        "first_convert_llama_physical_op_id": first_convert_llama_physical_op_id,
        "second_convert_logical_op_id": second_convert_logical_op_id,
        "second_convert_gpt4o_physical_op_id": second_convert_gpt4o_physical_op_id,
        "second_convert_gpt4o_mini_physical_op_id": second_convert_gpt4o_mini_physical_op_id,
        "second_convert_llama_physical_op_id": second_convert_llama_physical_op_id,
        "third_convert_logical_op_id": third_convert_logical_op_id,
        "third_convert_gpt4o_physical_op_id": third_convert_gpt4o_physical_op_id,
        "third_convert_gpt4o_mini_physical_op_id": third_convert_gpt4o_mini_physical_op_id,
        "third_convert_llama_physical_op_id": third_convert_llama_physical_op_id,
    }

@pytest.fixture
def three_converts_min_cost_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_three_converts_logical_and_physical_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_llama_physical_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_max_quality_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_three_converts_logical_and_physical_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_min_cost_at_fixed_quality_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_three_converts_logical_and_physical_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_physical_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_max_quality_at_fixed_cost_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_three_converts_logical_and_physical_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # normalize costs by cardinality; needs to cost less than 1.0 per record
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_physical_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_physical_op_id']: {"cost": 0.25 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

### ONE FILTER ONE CONVERT OPERATOR-TO-STATS ###
def get_one_filter_one_convert_logical_and_physical_op_ids(one_filter_one_convert_workload, enron_eval_tiny, email_schema):
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

    # get physical and logical op id for scan operator
    scan_physical_op_id = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny).get_op_id()
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()

    # get physical op ids for first filter operator
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_filter_gpt4o_physical_op_id = LLMFilter(output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    first_filter_gpt4o_mini_physical_op_id = LLMFilter(output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    first_filter_llama_physical_op_id = LLMFilter(output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    first_filter_logical_op = FilteredScan(input_schema=TextFile, output_schema=TextFile, filter=Filter("filter1"), depends_on=list(depends_on), target_cache_id=first_filter_node_id)
    first_filter_logical_op_id = first_filter_logical_op.get_logical_op_id()

    # get physical op ids for first convert operator
    depends_on = depends_on.union(set(first_filter_logical_op.output_schema.field_names(unique=True, id=first_filter_node_id)))
    first_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    first_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    first_convert_llama_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_physical_op_id": scan_physical_op_id,
        "first_filter_logical_op_id": first_filter_logical_op_id,
        "first_filter_gpt4o_physical_op_id": first_filter_gpt4o_physical_op_id,
        "first_filter_gpt4o_mini_physical_op_id": first_filter_gpt4o_mini_physical_op_id,
        "first_filter_llama_physical_op_id": first_filter_llama_physical_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_physical_op_id": first_convert_gpt4o_physical_op_id,
        "first_convert_gpt4o_mini_physical_op_id": first_convert_gpt4o_mini_physical_op_id,
        "first_convert_llama_physical_op_id": first_convert_llama_physical_op_id,
    }

@pytest.fixture
def one_filter_one_convert_min_cost_operator_to_stats(one_filter_one_convert_workload, enron_eval_tiny, email_schema):
    # get logical and physical op ids
    op_ids = get_one_filter_one_convert_logical_and_physical_op_ids(one_filter_one_convert_workload, enron_eval_tiny, email_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

### TWO CONVERTS TWO FILTERS OPERATOR-TO-STATS ###
def get_two_converts_two_filters_logical_and_physical_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # extract node id's from workload Datasets
    scan_node_id = two_converts_two_filters_workload._source._source._source._source.universal_identifier()
    first_convert_node_id = two_converts_two_filters_workload._source._source._source.universal_identifier()
    second_convert_node_id = two_converts_two_filters_workload._source._source.universal_identifier()
    first_filter_node_id = two_converts_two_filters_workload._source.universal_identifier()
    second_filter_node_id = two_converts_two_filters_workload.universal_identifier()

    # get physical and logical op id for scan operator
    scan_physical_op_id = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny).get_op_id()
    scan_logical_op = BaseScan(dataset_id=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()

    # get physical op ids for first convert operators
    depends_on = set(scan_logical_op.output_schema.field_names(unique=True, id=scan_node_id))
    first_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    first_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    first_convert_llama_physical_op_id = LLMConvertBonded(output_schema=email_schema, input_schema=TextFile, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on), target_cache_id=first_convert_node_id)
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()

    # get physical op ids for second convert operators
    depends_on.update(first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id))
    second_convert_gpt4o_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    second_convert_gpt4o_mini_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    second_convert_llama_physical_op_id = LLMConvertBonded(output_schema=foobar_schema, input_schema=email_schema, model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=foobar_schema, depends_on=list(depends_on), target_cache_id=second_convert_node_id)
    second_convert_logical_op_id = second_convert_logical_op.get_logical_op_id()

    # get physical op ids for first filter operators
    depends_on = [field for field in first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id) if "sender" in field]
    first_filter_gpt4o_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter1"), model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    first_filter_gpt4o_mini_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter1"), model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    first_filter_llama_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter1"), model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    first_filter_logical_op = FilteredScan(input_schema=foobar_schema, output_schema=foobar_schema, filter=Filter("filter1"), depends_on=list(depends_on), target_cache_id=first_filter_node_id)
    first_filter_logical_op_id = first_filter_logical_op.get_logical_op_id()

    # get physical op ids for second filter operators
    depends_on = [field for field in first_convert_logical_op.output_schema.field_names(unique=True, id=first_convert_node_id) if "subject" in field]
    second_filter_gpt4o_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter2"), model=Model.GPT_4o, depends_on=list(depends_on)).get_op_id()
    second_filter_gpt4o_mini_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter2"), model=Model.GPT_4o_MINI, depends_on=list(depends_on)).get_op_id()
    second_filter_llama_physical_op_id = LLMFilter(output_schema=foobar_schema, input_schema=foobar_schema, filter=Filter("filter2"), model=Model.LLAMA3, depends_on=list(depends_on)).get_op_id()
    second_filter_logical_op = FilteredScan(input_schema=foobar_schema, output_schema=foobar_schema, filter=Filter("filter2"), depends_on=list(depends_on), target_cache_id=second_filter_node_id)
    second_filter_logical_op_id = second_filter_logical_op.get_logical_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_physical_op_id": scan_physical_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_physical_op_id": first_convert_gpt4o_physical_op_id,
        "first_convert_gpt4o_mini_physical_op_id": first_convert_gpt4o_mini_physical_op_id,
        "first_convert_llama_physical_op_id": first_convert_llama_physical_op_id,
        "second_convert_logical_op_id": second_convert_logical_op_id,
        "second_convert_gpt4o_physical_op_id": second_convert_gpt4o_physical_op_id,
        "second_convert_gpt4o_mini_physical_op_id": second_convert_gpt4o_mini_physical_op_id,
        "second_convert_llama_physical_op_id": second_convert_llama_physical_op_id,
        "first_filter_logical_op_id": first_filter_logical_op_id,
        "first_filter_gpt4o_physical_op_id": first_filter_gpt4o_physical_op_id,
        "first_filter_gpt4o_mini_physical_op_id": first_filter_gpt4o_mini_physical_op_id,
        "first_filter_llama_physical_op_id": first_filter_llama_physical_op_id,
        "second_filter_logical_op_id": second_filter_logical_op_id,
        "second_filter_gpt4o_physical_op_id": second_filter_gpt4o_physical_op_id,
        "second_filter_gpt4o_mini_physical_op_id": second_filter_gpt4o_mini_physical_op_id,
        "second_filter_llama_physical_op_id": second_filter_llama_physical_op_id,
    }

@pytest.fixture
def two_converts_two_filters_min_cost_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_two_converts_two_filters_logical_and_physical_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
            op_ids['first_filter_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1 / 3},
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_max_quality_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_two_converts_two_filters_logical_and_physical_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.75},
            op_ids['first_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
            op_ids['first_filter_llama_physical_op_id']: {"cost": 0.75, "time": 1.0, "quality": 1.0, "selectivity": 0.75},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
            op_ids['second_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_filter_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_min_cost_at_fixed_quality_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_two_converts_two_filters_logical_and_physical_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 1st
            op_ids['first_convert_llama_physical_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 0.3, "time": 1.0, "quality": 1.0, "selectivity": 1.0}, # pick 4th
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": (1 / 3)},
            op_ids['first_filter_gpt4o_mini_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5}, # pick 2nd
            op_ids['first_filter_llama_physical_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_physical_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_gpt4o_mini_physical_op_id']: {"cost": 10.0, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
            op_ids['second_filter_llama_physical_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 0.75}, # pick 3rd
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_max_quality_at_fixed_cost_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and physical op ids
    op_ids = get_two_converts_two_filters_logical_and_physical_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # normalize costs by cardinality; needs to cost less than 1.0 per record
    cardinality = len(DataDirectory().get_registered_dataset(enron_eval_tiny))

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_physical_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_physical_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 1st
            op_ids['first_convert_llama_physical_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_physical_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_physical_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 4th
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0}, # pick 3rd
            op_ids['first_filter_gpt4o_mini_physical_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_filter_llama_physical_op_id']: {"cost": 0.1 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 0.5}, 
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_physical_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 0.5}, # pick 2nd
            op_ids['second_filter_gpt4o_mini_physical_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_filter_llama_physical_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
        },
    }

    return operator_to_stats
