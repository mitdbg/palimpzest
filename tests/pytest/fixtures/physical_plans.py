import pytest

from palimpzest.constants import Cardinality, Model
from palimpzest.core.data.datareaders import MemoryReader
from palimpzest.core.elements.filters import Filter
from palimpzest.core.lib.schemas import File, Schema, StringField, TextFile
from palimpzest.query.operators.code_synthesis_convert import CodeSynthesisConvertSingle
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.rag_convert import RAGConvert
from palimpzest.query.operators.scan import MarshalAndScanDataOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan

# from palimpzest.operators.token_reduction_convert import TokenReducedConvertBonded

### PHYSICAL PLANS ###
@pytest.fixture
def scan_only_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datareader=enron_eval_tiny)
    plan = PhysicalPlan(operators=[scan_op])
    return plan


@pytest.fixture
def non_llm_filter_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datareader=enron_eval_tiny)

    def filter_emails(record: dict):
        return record["filename"] in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]

    filter = Filter(filter_fn=filter_emails)
    filter_op = NonLLMFilter(input_schema=File, output_schema=File, filter=filter, target_cache_id="abc123")
    plan = PhysicalPlan(operators=[scan_op, filter_op])
    return plan


@pytest.fixture
def llm_filter_plan(enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=File, datareader=enron_eval_tiny)
    filter = Filter("This filter will be mocked out")
    filter_op = LLMFilter(
        input_schema=File,
        output_schema=File,
        filter=filter,
        model=Model.GPT_4o_MINI,
        target_cache_id="abc123",
    )
    plan = PhysicalPlan(operators=[scan_op, filter_op])
    return plan


@pytest.fixture
def bonded_llm_convert_plan(email_schema, enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datareader=enron_eval_tiny)
    convert_op_llm = LLMConvertBonded(
        input_schema=TextFile,
        output_schema=email_schema,
        model=Model.GPT_4o_MINI,
        target_cache_id="abc123",
    )
    plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def code_synth_convert_plan(email_schema, enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datareader=enron_eval_tiny)
    convert_op_llm = CodeSynthesisConvertSingle(
        input_schema=TextFile,
        output_schema=email_schema,
        exemplar_generation_model=Model.GPT_4o,
        code_synth_model=Model.GPT_4o,
        fallback_model=Model.GPT_4o_MINI,
        target_cache_id="abc123",
        cache_across_plans=False,
    )
    plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
    return plan

@pytest.fixture
def rag_convert_plan(email_schema, enron_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datareader=enron_eval_tiny)
    convert_op_llm = RAGConvert(
        input_schema=TextFile,
        output_schema=email_schema,
        model=Model.GPT_4o_MINI,
        num_chunks_per_field=1,
        chunk_size=1000,
        target_cache_id="abc123",
    )
    plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
    return plan

# NOTE: removing until TokenReducedConvertBonded has implementation changes
# @pytest.fixture
# def token_reduction_convert_plan(email_schema, enron_eval_tiny):
#     scan_op = MarshalAndScanDataOp(output_schema=TextFile, datareader=enron_eval_tiny)
#     convert_op_llm = TokenReducedConvertBonded(
#         input_schema=TextFile,
#         output_schema=email_schema,
#         model=Model.GPT_4o_MINI,
#         token_budget=0.1,
#         target_cache_id="abc123",
#     )
#     plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
#     return plan


@pytest.fixture
def image_convert_plan(real_estate_listing_files_schema, image_real_estate_listing_schema, real_estate_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=real_estate_listing_files_schema, datareader=real_estate_eval_tiny)
    convert_op_llm = LLMConvertBonded(
        input_schema=real_estate_listing_files_schema,
        output_schema=image_real_estate_listing_schema,
        model=Model.GPT_4o_MINI,
        target_cache_id="abc123",
    )
    plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def one_to_many_convert_plan(real_estate_listing_files_schema, room_real_estate_listing_schema, real_estate_eval_tiny):
    scan_op = MarshalAndScanDataOp(output_schema=real_estate_listing_files_schema, datareader=real_estate_eval_tiny)
    convert_op_llm = LLMConvertBonded(
        input_schema=real_estate_listing_files_schema,
        output_schema=room_real_estate_listing_schema,
        model=Model.GPT_4o_MINI,
        cardinality=Cardinality.ONE_TO_MANY,
        target_cache_id="abc123",
    )
    plan = PhysicalPlan(operators=[scan_op, convert_op_llm])
    return plan


@pytest.fixture
def simple_plan_factory():
    def simple_plan_generator(convert_model, filter_model):
        class FooSchema(Schema):
            foo = StringField("foo")

        datareader = MemoryReader(vals=[1, 2, 3, 4, 5, 6])
        scan_op = MarshalAndScanDataOp(output_schema=File, datareader=datareader)
        convert_op_llm = LLMConvertBonded(
            input_schema=File,
            output_schema=FooSchema,
            model=convert_model,
            target_cache_id="abc123",
        )
        filter = Filter("bar")
        filter_op = LLMFilter(
            input_schema=FooSchema,
            output_schema=FooSchema,
            filter=filter,
            model=filter_model,
            target_cache_id="abc123",
        )
        plan = PhysicalPlan(operators=[scan_op, convert_op_llm, filter_op])
        return plan

    return simple_plan_generator


@pytest.fixture
def scan_convert_filter_sentinel_plan(foobar_schema):
    datareader = MemoryReader(vals=[1, 2, 3, 4, 5, 6])
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, logical_op_id="scan1", datareader=datareader)
    convert_ops = [
        LLMConvertBonded(
            input_schema=TextFile,
            output_schema=foobar_schema,
            model=model,
            logical_op_id="convert1",
            target_cache_id=f"convert-foobar-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    filter_ops = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("hello"),
            model=model,
            logical_op_id="filter1",
            target_cache_id=f"filter-hello-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    plan = SentinelPlan(operator_sets=[[scan_op], convert_ops, filter_ops])
    return plan


@pytest.fixture
def scan_multi_convert_multi_filter_sentinel_plan(foobar_schema, baz_schema):
    datareader = MemoryReader(vals=[1, 2, 3, 4, 5, 6])
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, logical_op_id="scan1", datareader=datareader)
    convert_ops1 = [
        LLMConvertBonded(
            input_schema=TextFile,
            output_schema=foobar_schema,
            model=model,
            logical_op_id="convert1",
            target_cache_id=f"convert-foobar-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    filter_ops1 = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("hello"),
            model=model,
            logical_op_id="filter1",
            target_cache_id=f"filter-hello-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    filter_ops2 = [
        LLMFilter(
            input_schema=foobar_schema,
            output_schema=foobar_schema,
            filter=Filter("world"),
            model=model,
            logical_op_id="filter2",
            target_cache_id=f"filter-world-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    convert_ops2 = [
        LLMConvertBonded(
            input_schema=foobar_schema,
            output_schema=baz_schema,
            model=model,
            logical_op_id="convert2",
            target_cache_id=f"convert-baz-{model.value}",
        )
        for model in [Model.GPT_4o_MINI, Model.GPT_4o, Model.MIXTRAL]
    ]
    plan = SentinelPlan(operator_sets=[[scan_op], convert_ops1, filter_ops1, filter_ops2, convert_ops2])
    return plan
