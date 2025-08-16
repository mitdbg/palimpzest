import pytest

from palimpzest.constants import Cardinality, Model
from palimpzest.core.data.iter_dataset import MemoryDataset
from palimpzest.core.elements.filters import Filter
from palimpzest.core.lib.schemas import File, TextFile
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.rag_convert import RAGConvert
from palimpzest.query.operators.scan import MarshalAndScanDataOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan


### PHYSICAL PLANS ###
@pytest.fixture
def scan_only_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datasource=enron_eval_tiny, logical_op_id="scan1")
    plan = PhysicalPlan._from_ops(ops=[scan_op])
    return plan


@pytest.fixture
def non_llm_filter_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datasource=enron_eval_tiny, logical_op_id="scan1")

    def filter_emails(record: dict):
        return record["filename"] in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]

    filter = Filter(filter_fn=filter_emails)
    filter_op = NonLLMFilter(input_schema=File, output_schema=File, filter=filter, logical_op_id="filter1")
    plan = PhysicalPlan._from_ops(ops=[scan_op, filter_op])
    return plan


@pytest.fixture
def llm_filter_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datasource=enron_eval_tiny, logical_op_id="scan1")
    filter = Filter("This filter will be mocked out")
    filter_op = LLMFilter(
        input_schema=File,
        output_schema=File,
        filter=filter,
        model=Model.GPT_4o_MINI,
        logical_op_id="filter1",
    )
    plan = PhysicalPlan._from_ops(ops=[scan_op, filter_op])
    return plan


@pytest.fixture
def bonded_llm_convert_plan(email_schema, enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datasource=enron_eval_tiny, logical_op_id="scan1")
    convert_op_llm = LLMConvertBonded(
        input_schema=TextFile,
        output_schema=email_schema,
        model=Model.GPT_4o_MINI,
        logical_op_id="convert1",
    )
    plan = PhysicalPlan._from_ops(ops=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def rag_convert_plan(email_schema, enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datasource=enron_eval_tiny, logical_op_id="scan1")
    convert_op_llm = RAGConvert(
        input_schema=TextFile,
        output_schema=email_schema,
        model=Model.GPT_4o_MINI,
        num_chunks_per_field=1,
        chunk_size=1000,
        logical_op_id="rag_convert1",
    )
    plan = PhysicalPlan._from_ops(ops=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def image_convert_plan(real_estate_listing_files_schema, image_real_estate_listing_schema, real_estate_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=real_estate_listing_files_schema, datasource=real_estate_eval_tiny, logical_op_id="scan1")
    convert_op_llm = LLMConvertBonded(
        input_schema=real_estate_listing_files_schema,
        output_schema=image_real_estate_listing_schema,
        model=Model.GPT_4o_MINI,
        logical_op_id="convert1",
    )
    plan = PhysicalPlan._from_ops(ops=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def one_to_many_convert_plan(real_estate_listing_files_schema, room_real_estate_listing_schema, real_estate_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=real_estate_listing_files_schema, datasource=real_estate_eval_tiny, logical_op_id="scan1")
    convert_op_llm = LLMConvertBonded(
        input_schema=real_estate_listing_files_schema,
        output_schema=room_real_estate_listing_schema,
        model=Model.GPT_4o_MINI,
        cardinality=Cardinality.ONE_TO_MANY,
        logical_op_id="convert1",
    )
    plan = PhysicalPlan._from_ops(ops=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def scan_convert_filter_sentinel_plan(foobar_schema):
    datasource = MemoryDataset(id="test", vals=[1, 2, 3, 4, 5, 6])
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datasource=datasource, logical_op_id="scan1")
    convert_ops = [
        LLMConvertBonded(
            input_schema=TextFile,
            output_schema=foobar_schema,
            model=model,
            logical_op_id="convert1",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    filter_ops = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("hello"),
            model=model,
            logical_op_id="filter1",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    plan = SentinelPlan(operator_sets=[[scan_op], convert_ops, filter_ops])
    return plan


@pytest.fixture
def scan_multi_convert_multi_filter_sentinel_plan(foobar_schema, baz_schema):
    datasource = MemoryDataset(id="test", vals=[1, 2, 3, 4, 5, 6])
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datasource=datasource, logical_op_id="scan1")
    convert_ops1 = [
        LLMConvertBonded(
            input_schema=TextFile,
            output_schema=foobar_schema,
            model=model,
            logical_op_id="convert1",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    filter_ops1 = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("hello"),
            model=model,
            logical_op_id="filter1",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    filter_ops2 = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("world"),
            model=model,
            logical_op_id="filter2",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    convert_ops2 = [
        LLMConvertBonded(
            input_schema=foobar_schema,
            output_schema=baz_schema,
            model=model,
            logical_op_id="convert2",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.LLAMA3_1_8B]
    ]
    plan = SentinelPlan(operator_sets=[[scan_op], convert_ops1, filter_ops1, filter_ops2, convert_ops2])
    return plan
