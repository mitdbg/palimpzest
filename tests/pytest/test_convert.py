"""This testing class is an integration test suite.
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import os
import sys

import pytest

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")

from palimpzest.constants import Model, PromptStrategy
from palimpzest.corelib.schemas import File, TextFile
from palimpzest.datamanager import DataDirectory
from palimpzest.elements.records import DataRecord
from palimpzest.operators import LLMConvertBonded, LLMConvertConventional
from palimpzest.operators.datasource import MarshalAndScanDataOp

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils import load_env

    load_env()


@pytest.mark.parametrize("convert_op", [LLMConvertBonded, LLMConvertConventional])
def test_convert(convert_op, email_schema, enron_eval_tiny):
    """Test whether convert operators"""
    model = Model.GPT_4
    scanOp = MarshalAndScanDataOp(outputSchema=TextFile, dataset_id=enron_eval_tiny)
    convertOp = convert_op(
        inputSchema=File,
        outputSchema=email_schema,
        model=model,
        prompt_strategy=PromptStrategy.DSPY_COT_QA,
    )

    datasource = DataDirectory().getRegisteredDataset(enron_eval_tiny)
    candidate = DataRecord(schema=File, parent_id=None, scan_idx=0)
    candidate.idx = 0
    candidate.get_item_fn = datasource.getItem
    candidate.cardinality = datasource.cardinality
    # run DataSourcePhysicalOp on record

    outputs = []
    records, _ = scanOp(candidate)
    for record in records:
        output, _ = convertOp(record)
        outputs.extend(output)

    for record in outputs:
        print(record.sender, record.subject)
