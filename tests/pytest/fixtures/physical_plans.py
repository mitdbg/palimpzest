import pytest
import palimpzest as pz
from palimpzest.corelib import File
from palimpzest.operators import *
from palimpzest.optimizer import PhysicalPlan

### PHYSICAL PLANS ###
# TODO: provide dataset_id as argument to these fixtures
@pytest.fixture
def scan_only_plan():
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_id="enron-eval-tiny")
    plan = PhysicalPlan(operators=[scanOp])
    return plan

@pytest.fixture
def non_llm_filter_plan():
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_id="enron-eval-tiny")
    def filter_emails(record):
        return record.filename in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]
    filter = Filter(filterFn=filter_emails)
    filterOp = NonLLMFilter(inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123")
    plan = PhysicalPlan(operators=[scanOp, filterOp])
    return plan

@pytest.fixture
def llm_filter_plan():
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_id="enron-eval-tiny")
    filter = Filter("This filter will be mocked out")
    filterOp = LLMFilter(
        inputSchema=File,
        outputSchema=File,
        filter=filter,
        model=Model.GPT_3_5,
        targetCacheId="abc123",
    )
    plan = PhysicalPlan(operators=[scanOp, filterOp])
    return plan

@pytest.fixture
def bonded_llm_convert_plan(email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=TextFile, dataset_id="enron-eval-tiny")
    convertOpLLM = LLMConvertBonded(
        inputSchema=TextFile,
        outputSchema=email_schema,
        model=Model.GPT_3_5,
        targetCacheId="abc123",
    )
    plan = PhysicalPlan(operators=[scanOp, convertOpLLM])
    return plan

@pytest.fixture
def code_synth_convert_plan(email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=TextFile, dataset_id="enron-eval-tiny")
    convertOpLLM = CodeSynthesisConvertSingle(
        inputSchema=TextFile,
        outputSchema=email_schema,
        exemplar_generation_model=Model.GPT_4,
        code_synth_model=Model.GPT_4,
        conventional_fallback_model=Model.GPT_3_5,
        targetCacheId="abc123",
        cache_across_plans=False,
    )
    plan = PhysicalPlan(operators=[scanOp, convertOpLLM])
    return plan

@pytest.fixture
def token_reduction_convert_plan(email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=TextFile, dataset_id="enron-eval-tiny")
    convertOpLLM = TokenReducedConvertBonded(
        inputSchema=TextFile,
        outputSchema=email_schema,
        model=Model.GPT_3_5,
        token_budget=0.1,
        targetCacheId="abc123",
    )
    plan = PhysicalPlan(operators=[scanOp, convertOpLLM])
    return plan

@pytest.fixture
def image_convert_plan(real_estate_listing_files_schema, image_real_estate_listing_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=real_estate_listing_files_schema, dataset_id="real-estate-eval-tiny")
    convertOpLLM = LLMConvertBonded(
        inputSchema=real_estate_listing_files_schema,
        outputSchema=image_real_estate_listing_schema,
        model=Model.GPT_3_5,
        targetCacheId="abc123",
        image_conversion=True,
    )
    plan = PhysicalPlan(operators=[scanOp, convertOpLLM])
    return plan

@pytest.fixture
def one_to_many_convert_plan(real_estate_listing_files_schema, room_real_estate_listing_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=real_estate_listing_files_schema, dataset_id="real-estate-eval-tiny")
    convertOpLLM = LLMConvertBonded(
        inputSchema=real_estate_listing_files_schema,
        outputSchema=room_real_estate_listing_schema,
        model=Model.GPT_3_5,
        cardinality=Cardinality.ONE_TO_MANY,
        targetCacheId="abc123",
        image_conversion=True,
    )
    plan = PhysicalPlan(operators=[scanOp, convertOpLLM])
    return plan


@pytest.fixture
def simple_plan_factory():
    def simple_plan_generator(convert_model, filter_model):
        class FooSchema(pz.Schema):
            foo = pz.StringField("foo")

        scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_id="foobar")
        convertOpLLM = LLMConvertBonded(
            inputSchema=File,
            outputSchema=FooSchema,
            model=convert_model,
            targetCacheId="abc123",
        )
        filter = Filter("bar")
        filterOp = LLMFilter(
            inputSchema=FooSchema,
            outputSchema=FooSchema,
            filter=filter,
            model=filter_model,
            targetCacheId="abc123",
        )
        plan = PhysicalPlan(operators=[scanOp, convertOpLLM, filterOp])
        return plan

    return simple_plan_generator
