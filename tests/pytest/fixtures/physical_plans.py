import pytest
from palimpzest.corelib import File
from palimpzest.operators import *
from palimpzest.planner import PhysicalPlan
from palimpzest.strategies import (
    BondedQueryConvertStrategy,
    CodeSynthesisConvertStrategy,
    ModelSelectionFilterStrategy,
    TokenReducedConvertStrategy,
)

### PHYSICAL PLANS ###
@pytest.fixture
def enron_scan_only_plan(enron_eval_tiny):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    plan = PhysicalPlan(
        operators=[scanOp],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def enron_non_llm_filter_plan(enron_eval_tiny):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    def filter_emails(record):
        return record.filename in ["buy-r-inbox-628.txt", "buy-r-inbox-749.txt", "zipper-a-espeed-28.txt"]
    filter = Filter(filterFn=filter_emails)
    filterOp = NonLLMFilter(inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True)
    plan = PhysicalPlan(
        operators=[scanOp, filterOp],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def enron_llm_filter_plan(enron_eval_tiny):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    filter = Filter("This filter will be mocked out")
    filterOpClass = ModelSelectionFilterStrategy(available_models=[Model.GPT_3_5], prompt_strategy=PromptStrategy.DSPY_COT_BOOL)[0]
    filterOp = filterOpClass(inputSchema=File, outputSchema=File, filter=filter, targetCacheId="abc123", shouldProfile=True)
    plan = PhysicalPlan(
        operators=[scanOp, filterOp],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def enron_bonded_llm_convert_plan(enron_eval_tiny, email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
    convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]
    convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True)
    plan = PhysicalPlan(
        operators=[scanOp, convertOpHardcoded, convertOpLLM],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def enron_code_synth_convert_plan(enron_eval_tiny, email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
    convertOpClass = CodeSynthesisConvertStrategy(code_synth_strategy=CodingStrategy.SINGLE)[0]
    convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True, cache_across_plans=False)
    plan = PhysicalPlan(
        operators=[scanOp, convertOpHardcoded, convertOpLLM],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def enron_token_reduction_convert_plan(enron_eval_tiny, email_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=File, dataset_type="dir", shouldProfile=True)
    convertOpHardcoded = ConvertFileToText(inputSchema=File, outputSchema=TextFile, shouldProfile=True)
    # using [1] index to get bonded query convert
    convertOpClass = TokenReducedConvertStrategy(available_models=[Model.GPT_3_5], token_budgets=[0.1])[1]
    convertOpLLM = convertOpClass(inputSchema=TextFile, outputSchema=email_schema, targetCacheId="abc123", shouldProfile=True)
    plan = PhysicalPlan(
        operators=[scanOp, convertOpHardcoded, convertOpLLM],
        datasetIdentifier=enron_eval_tiny,
    )
    return plan

@pytest.fixture
def real_estate_image_convert_plan(real_estate_eval_tiny, real_estate_listing_files_schema, image_real_estate_listing_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=real_estate_listing_files_schema, dataset_type="dir", shouldProfile=True)
    convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]
    convertOpLLM = convertOpClass(inputSchema=real_estate_listing_files_schema, outputSchema=image_real_estate_listing_schema, targetCacheId="abc123", shouldProfile=True, image_conversion=True)
    plan = PhysicalPlan(
        operators=[scanOp, convertOpLLM],
        datasetIdentifier=real_estate_eval_tiny,
    )
    return plan

@pytest.fixture
def real_estate_one_to_many_convert_plan(real_estate_eval_tiny, real_estate_listing_files_schema, room_real_estate_listing_schema):
    scanOp = MarshalAndScanDataOp(outputSchema=real_estate_listing_files_schema, dataset_type="dir", shouldProfile=True)
    convertOpClass = BondedQueryConvertStrategy(available_models=[Model.GPT_3_5])[0]
    convertOpLLM = convertOpClass(inputSchema=real_estate_listing_files_schema, outputSchema=room_real_estate_listing_schema, cardinality=Cardinality.ONE_TO_MANY, targetCacheId="abc123", shouldProfile=True, image_conversion=True)
    plan = PhysicalPlan(
        operators=[scanOp, convertOpLLM],
        datasetIdentifier=real_estate_eval_tiny,
    )
    return plan
