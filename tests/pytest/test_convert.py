"""This testing class is an integration test suite.
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import os
import sys

import pytest

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import File, TextFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.query.operators.convert import LLMConvertBonded, LLMConvertConventional
from palimpzest.query.operators.datasource import MarshalAndScanDataOp

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


@pytest.mark.parametrize(
    argnames=("convert_op", "side_effect"),
    argvalues=[
        pytest.param(LLMConvertBonded, "enron-convert", id="bonded-llm-convert"),
        pytest.param(LLMConvertConventional, "enron-convert", id="conventional-llm-convert"),
    ],
    indirect=["side_effect"],
)
def test_convert(mocker, convert_op, side_effect, email_schema, enron_eval_tiny):
    """Test whether convert operators"""
    model = Model.GPT_4o
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, dataset_id=enron_eval_tiny)
    convert_op = convert_op(
        input_schema=File,
        output_schema=email_schema,
        model=model,
        prompt_strategy=PromptStrategy.COT_QA,
    )

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=side_effect)
    mocker.patch.object(LLMConvertConventional, "convert", side_effect=side_effect)
 
    datasource = DataDirectory().get_registered_dataset(enron_eval_tiny)
    candidate = DataRecord(schema=File, source_id=0)
    candidate.idx = 0
    candidate.get_item_fn = datasource.get_item

    # run scan and convert operators
    record_op_stats_lst, outputs = [], []
    for record in scan_op(candidate):
        record_set = convert_op(record)
        record_op_stats_lst.extend(record_set.record_op_stats)
        outputs.extend(record_set.data_records)

    assert len(outputs) == 1
    assert outputs[0].schema == email_schema.union(TextFile)
    assert sorted(outputs[0].get_field_names()) == ["contents", "filename", "sender", "subject"]
