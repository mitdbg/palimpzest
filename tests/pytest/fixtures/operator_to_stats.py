from copy import deepcopy

import pytest

from palimpzest.constants import CuratedModel
from palimpzest.utils.model_info import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.lib.schemas import TextFile, get_schema_field_names, union_schemas
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.logical import BaseScan, ConvertScan, FilteredScan
from palimpzest.query.operators.scan import MarshalAndScanDataOp


### THREE CONVERTS OPERATOR-TO-STATS ###
def get_three_converts_logical_and_full_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # extract node id's from workload Datasets
    scan_node_id = three_converts_workload._sources[0]._sources[0]._sources[0].id
    first_convert_node_id = three_converts_workload._sources[0]._sources[0].id
    second_convert_node_id = three_converts_workload._sources[0].id
    # third_convert_node_id = three_converts_workload.id

    # get full and logical op id for scan operator
    scan_logical_op = BaseScan(datasource=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()
    scan_full_op_id = MarshalAndScanDataOp(logical_op_id=scan_logical_op_id, output_schema=TextFile, datasource=enron_eval_tiny).get_full_op_id()

    # get full op ids for first convert operators
    depends_on = set(get_schema_field_names(scan_logical_op.output_schema, id=scan_node_id))
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on))
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()
    first_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    first_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    first_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for second convert operators
    depends_on.update(get_schema_field_names(first_convert_logical_op.output_schema, id=first_convert_node_id))
    second_output_schema = union_schemas([email_schema, foobar_schema])
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=second_output_schema, depends_on=list(depends_on))
    second_convert_logical_op_id = second_convert_logical_op.get_logical_op_id()
    second_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=second_output_schema, input_schema=email_schema, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    second_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=second_output_schema, input_schema=email_schema, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    second_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=second_output_schema, input_schema=email_schema, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for third convert operators
    depends_on.update(get_schema_field_names(second_convert_logical_op.output_schema, id=second_convert_node_id))
    third_output_schema = union_schemas([second_output_schema, baz_schema])
    third_convert_logical_op = ConvertScan(input_schema=second_output_schema, output_schema=third_output_schema, depends_on=list(depends_on))
    third_convert_logical_op_id = third_convert_logical_op.get_logical_op_id()
    third_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=third_convert_logical_op_id, output_schema=third_output_schema, input_schema=second_output_schema, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    third_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=third_convert_logical_op_id, output_schema=third_output_schema, input_schema=second_output_schema, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    third_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=third_convert_logical_op_id, output_schema=third_output_schema, input_schema=second_output_schema, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_full_op_id": scan_full_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_full_op_id": first_convert_gpt4o_full_op_id,
        "first_convert_gpt4o_mini_full_op_id": first_convert_gpt4o_mini_full_op_id,
        "first_convert_llama_full_op_id": first_convert_llama_full_op_id,
        "second_convert_logical_op_id": second_convert_logical_op_id,
        "second_convert_gpt4o_full_op_id": second_convert_gpt4o_full_op_id,
        "second_convert_gpt4o_mini_full_op_id": second_convert_gpt4o_mini_full_op_id,
        "second_convert_llama_full_op_id": second_convert_llama_full_op_id,
        "third_convert_logical_op_id": third_convert_logical_op_id,
        "third_convert_gpt4o_full_op_id": third_convert_gpt4o_full_op_id,
        "third_convert_gpt4o_mini_full_op_id": third_convert_gpt4o_mini_full_op_id,
        "third_convert_llama_full_op_id": third_convert_llama_full_op_id,
    }

@pytest.fixture
def three_converts_min_cost_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_three_converts_logical_and_full_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_llama_full_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_max_quality_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_three_converts_logical_and_full_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_min_cost_at_fixed_quality_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_three_converts_logical_and_full_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_full_op_id']: {"cost": 0.3, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def three_converts_max_quality_at_fixed_cost_operator_to_stats(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_three_converts_logical_and_full_op_ids(three_converts_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # normalize costs by cardinality; needs to cost less than 1.0 per record
    cardinality = len(enron_eval_tiny)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['third_convert_logical_op_id']: {
            op_ids['third_convert_gpt4o_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['third_convert_gpt4o_mini_full_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['third_convert_llama_full_op_id']: {"cost": 0.25 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

### ONE FILTER ONE CONVERT OPERATOR-TO-STATS ###
def get_one_filter_one_convert_logical_and_full_op_ids(one_filter_one_convert_workload, enron_eval_tiny, email_schema):
    dataset_nodes = []
    node = deepcopy(one_filter_one_convert_workload)
    while not node.is_root:
        dataset_nodes.append(node)
        node = node._sources[0]
    dataset_nodes.append(node)
    dataset_nodes = list(reversed(dataset_nodes))

    # extract node id's from workload Datasets
    scan_node_id = dataset_nodes[0].id
    first_filter_node_id = dataset_nodes[1].id
    # first_convert_node_id = dataset_nodes[2].id

    # get full and logical op id for scan operator
    scan_logical_op = BaseScan(datasource=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()
    scan_full_op_id = MarshalAndScanDataOp(logical_op_id=scan_logical_op_id, output_schema=TextFile, datasource=enron_eval_tiny).get_full_op_id()

    # get full op ids for first filter operator
    depends_on = set(get_schema_field_names(scan_logical_op.output_schema, id=scan_node_id))
    first_filter_logical_op = FilteredScan(input_schema=TextFile, output_schema=TextFile, filter=Filter("filter1"), depends_on=list(depends_on))
    first_filter_logical_op_id = first_filter_logical_op.get_logical_op_id()
    first_filter_gpt4o_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    first_filter_gpt4o_mini_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    first_filter_llama_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=TextFile, input_schema=TextFile, filter=Filter("filter1"), model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for first convert operator
    depends_on = depends_on.union(set(get_schema_field_names(first_filter_logical_op.output_schema, id=first_filter_node_id)))
    output_schema = union_schemas([TextFile, email_schema])
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=output_schema, depends_on=list(depends_on))
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()
    first_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=output_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    first_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=output_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    first_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=output_schema, input_schema=TextFile, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_full_op_id": scan_full_op_id,
        "first_filter_logical_op_id": first_filter_logical_op_id,
        "first_filter_gpt4o_full_op_id": first_filter_gpt4o_full_op_id,
        "first_filter_gpt4o_mini_full_op_id": first_filter_gpt4o_mini_full_op_id,
        "first_filter_llama_full_op_id": first_filter_llama_full_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_full_op_id": first_convert_gpt4o_full_op_id,
        "first_convert_gpt4o_mini_full_op_id": first_convert_gpt4o_mini_full_op_id,
        "first_convert_llama_full_op_id": first_convert_llama_full_op_id,
    }

@pytest.fixture
def one_filter_one_convert_min_cost_operator_to_stats(one_filter_one_convert_workload, enron_eval_tiny, email_schema):
    # get logical and full op ids
    op_ids = get_one_filter_one_convert_logical_and_full_op_ids(one_filter_one_convert_workload, enron_eval_tiny, email_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
    }

    return operator_to_stats

### TWO CONVERTS TWO FILTERS OPERATOR-TO-STATS ###
def get_two_converts_two_filters_logical_and_full_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # extract node id's from workload Datasets
    scan_node_id = two_converts_two_filters_workload._sources[0]._sources[0]._sources[0]._sources[0].id
    first_convert_node_id = two_converts_two_filters_workload._sources[0]._sources[0]._sources[0].id
    # second_convert_node_id = two_converts_two_filters_workload._sources[0]._sources[0].id
    # first_filter_node_id = two_converts_two_filters_workload._sources[0].id
    # second_filter_node_id = two_converts_two_filters_workload.id

    # get full and logical op id for scan operator
    scan_logical_op = BaseScan(datasource=enron_eval_tiny, output_schema=TextFile)
    scan_logical_op_id = scan_logical_op.get_logical_op_id()
    scan_full_op_id = MarshalAndScanDataOp(logical_op_id=scan_logical_op_id, output_schema=TextFile, datasource=enron_eval_tiny).get_full_op_id()

    # get full op ids for first convert operators
    depends_on = set(get_schema_field_names(scan_logical_op.output_schema, id=scan_node_id))
    first_convert_logical_op = ConvertScan(input_schema=TextFile, output_schema=email_schema, depends_on=list(depends_on))
    first_convert_logical_op_id = first_convert_logical_op.get_logical_op_id()
    first_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    first_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    first_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=first_convert_logical_op_id, output_schema=email_schema, input_schema=TextFile, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for second convert operators
    depends_on.update(get_schema_field_names(first_convert_logical_op.output_schema, id=first_convert_node_id))
    output_schema = union_schemas([email_schema, foobar_schema])
    second_convert_logical_op = ConvertScan(input_schema=email_schema, output_schema=output_schema, depends_on=list(depends_on))
    second_convert_logical_op_id = second_convert_logical_op.get_logical_op_id()
    second_convert_gpt4o_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=output_schema, input_schema=email_schema, model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    second_convert_gpt4o_mini_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=output_schema, input_schema=email_schema, model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    second_convert_llama_full_op_id = LLMConvertBonded(logical_op_id=second_convert_logical_op_id, output_schema=output_schema, input_schema=email_schema, model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for first filter operators
    depends_on = [field for field in get_schema_field_names(first_convert_logical_op.output_schema, id=first_convert_node_id) if "sender" in field]
    first_filter_logical_op = FilteredScan(input_schema=output_schema, output_schema=output_schema, filter=Filter("filter1"), depends_on=list(depends_on))
    first_filter_logical_op_id = first_filter_logical_op.get_logical_op_id()
    first_filter_gpt4o_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter1"), model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    first_filter_gpt4o_mini_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter1"), model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    first_filter_llama_full_op_id = LLMFilter(logical_op_id=first_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter1"), model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    # get full op ids for second filter operators
    depends_on = [field for field in get_schema_field_names(first_convert_logical_op.output_schema, id=first_convert_node_id) if "subject" in field]
    second_filter_logical_op = FilteredScan(input_schema=output_schema, output_schema=output_schema, filter=Filter("filter2"), depends_on=list(depends_on))
    second_filter_logical_op_id = second_filter_logical_op.get_logical_op_id()
    second_filter_gpt4o_full_op_id = LLMFilter(logical_op_id=second_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter2"), model=Model(CuratedModel.GPT_4o), depends_on=list(depends_on)).get_full_op_id()
    second_filter_gpt4o_mini_full_op_id = LLMFilter(logical_op_id=second_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter2"), model=Model(CuratedModel.GPT_4o_MINI), depends_on=list(depends_on)).get_full_op_id()
    second_filter_llama_full_op_id = LLMFilter(logical_op_id=second_filter_logical_op_id, output_schema=output_schema, input_schema=output_schema, filter=Filter("filter2"), model=Model(CuratedModel.LLAMA3_3_70B), depends_on=list(depends_on)).get_full_op_id()

    return {
        "scan_logical_op_id": scan_logical_op_id,
        "scan_full_op_id": scan_full_op_id,
        "first_convert_logical_op_id": first_convert_logical_op_id,
        "first_convert_gpt4o_full_op_id": first_convert_gpt4o_full_op_id,
        "first_convert_gpt4o_mini_full_op_id": first_convert_gpt4o_mini_full_op_id,
        "first_convert_llama_full_op_id": first_convert_llama_full_op_id,
        "second_convert_logical_op_id": second_convert_logical_op_id,
        "second_convert_gpt4o_full_op_id": second_convert_gpt4o_full_op_id,
        "second_convert_gpt4o_mini_full_op_id": second_convert_gpt4o_mini_full_op_id,
        "second_convert_llama_full_op_id": second_convert_llama_full_op_id,
        "first_filter_logical_op_id": first_filter_logical_op_id,
        "first_filter_gpt4o_full_op_id": first_filter_gpt4o_full_op_id,
        "first_filter_gpt4o_mini_full_op_id": first_filter_gpt4o_mini_full_op_id,
        "first_filter_llama_full_op_id": first_filter_llama_full_op_id,
        "second_filter_logical_op_id": second_filter_logical_op_id,
        "second_filter_gpt4o_full_op_id": second_filter_gpt4o_full_op_id,
        "second_filter_gpt4o_mini_full_op_id": second_filter_gpt4o_mini_full_op_id,
        "second_filter_llama_full_op_id": second_filter_llama_full_op_id,
    }

@pytest.fixture
def two_converts_two_filters_min_cost_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_two_converts_two_filters_logical_and_full_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.1, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
            op_ids['first_filter_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1 / 3},
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_max_quality_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_two_converts_two_filters_logical_and_full_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.8, "selectivity": 1.0},
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.75},
            op_ids['first_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
            op_ids['first_filter_llama_full_op_id']: {"cost": 0.75, "time": 1.0, "quality": 1.0, "selectivity": 0.75},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5},
            op_ids['second_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_filter_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_min_cost_at_fixed_quality_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_two_converts_two_filters_logical_and_full_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 1st
            op_ids['first_convert_llama_full_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 0.3, "time": 1.0, "quality": 1.0, "selectivity": 1.0}, # pick 4th
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": (1 / 3)},
            op_ids['first_filter_gpt4o_mini_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 1.0, "selectivity": 0.5}, # pick 2nd
            op_ids['first_filter_llama_full_op_id']: {"cost": 1.0, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_full_op_id']: {"cost": 10.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_filter_gpt4o_mini_full_op_id']: {"cost": 10.0, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
            op_ids['second_filter_llama_full_op_id']: {"cost": 0.5, "time": 1.0, "quality": 0.9, "selectivity": 0.75}, # pick 3rd
        },
    }

    return operator_to_stats

@pytest.fixture
def two_converts_two_filters_max_quality_at_fixed_cost_operator_to_stats(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema):
    # get logical and full op ids
    op_ids = get_two_converts_two_filters_logical_and_full_op_ids(two_converts_two_filters_workload, enron_eval_tiny, email_schema, foobar_schema, baz_schema)

    # normalize costs by cardinality; needs to cost less than 1.0 per record
    cardinality = len(enron_eval_tiny)

    # construct operator_to_stats
    operator_to_stats = {
        op_ids['scan_logical_op_id']: {
            op_ids['scan_full_op_id']: {"cost": 0.0, "time": 1.0, "quality": 1.0, "selectivity": 1.0}
        },
        op_ids['first_convert_logical_op_id']: {
            op_ids['first_convert_gpt4o_full_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['first_convert_gpt4o_mini_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 1st
            op_ids['first_convert_llama_full_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
        },
        op_ids['second_convert_logical_op_id']: {
            op_ids['second_convert_gpt4o_full_op_id']: {"cost": 2.0 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0},
            op_ids['second_convert_gpt4o_mini_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_convert_llama_full_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0}, # pick 4th
        },
        op_ids['first_filter_logical_op_id']: {
            op_ids['first_filter_gpt4o_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 1.0}, # pick 3rd
            op_ids['first_filter_gpt4o_mini_full_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['first_filter_llama_full_op_id']: {"cost": 0.1 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 0.5}, 
        },
        op_ids['second_filter_logical_op_id']: {
            op_ids['second_filter_gpt4o_full_op_id']: {"cost": 0.3 / cardinality, "time": 1.0, "quality": 1.0, "selectivity": 0.5}, # pick 2nd
            op_ids['second_filter_gpt4o_mini_full_op_id']: {"cost": 0.5 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 1.0},
            op_ids['second_filter_llama_full_op_id']: {"cost": 0.2 / cardinality, "time": 1.0, "quality": 0.9, "selectivity": 0.5},
        },
    }

    return operator_to_stats
