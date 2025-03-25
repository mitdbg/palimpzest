"""This testing class is an integration test suite.
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import os

import pytest

# sys.path.append("./tests/")
# sys.path.append("./tests/refactor-tests/")
from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.lib.schemas import File, TextFile
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.scan import MarshalAndScanDataOp

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


@pytest.mark.parametrize(
    argnames=("convert_op", "side_effect"),
    argvalues=[
        pytest.param(LLMConvertBonded, "enron-convert", id="bonded-llm-convert"),
    ],
    indirect=["side_effect"],
)
def test_convert(mocker, convert_op, side_effect, email_schema, enron_eval_tiny):
    """Test whether convert operators"""
    scan_op = MarshalAndScanDataOp(output_schema=TextFile, datareader=enron_eval_tiny)
    convert_op = convert_op(
        input_schema=File,
        output_schema=email_schema,
        model=Model.GPT_4o,
        prompt_strategy=PromptStrategy.COT_QA,
    )

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=side_effect)

    # run scan and convert operators
    source_idx = 0
    record_op_stats_lst, outputs = [], []
    for record in scan_op(source_idx):
        record_set = convert_op(record)
        record_op_stats_lst.extend(record_set.record_op_stats)
        outputs.extend(record_set.data_records)

    assert len(outputs) == 1
    assert outputs[0].schema == email_schema.union(TextFile)
    assert sorted(outputs[0].get_field_names()) == ["contents", "filename", "sender", "subject"]
