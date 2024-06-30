""" This testing class is an integration test suite. 
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import sys
import pdb

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")
import context

import palimpzest as pz
from palimpzest.planner import LogicalPlanner, PhysicalPlanner
from palimpzest.operators import ConvertOp, ConvertFileToText
from palimpzest.execution import Execute
from utils import remove_cache, buildNestedStr
from palimpzest.strategies.model_selection import ModelSelectionFilterStrategy
from palimpzest.execution import SimpleExecution
from palimpzest.execution.nosentinel_execution import NoSentinelExecution
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.elements.records import DataRecord
from palimpzest.operators.convert import LLMConvertConventional
from palimpzest.strategies.bonded_query import LLMBondedQueryConvert
from palimpzest.operators.datasource import MarshalAndScanDataOp
from palimpzest.constants import PromptStrategy


def test_strategy(email_schema):
    """Test whether strategy creation works"""
    available_models = [pz.Model.GPT_4]

    emails = pz.Dataset("enron-eval-tiny", schema=email_schema)
    records, plan, stats= Execute(emails, 
                                  policy=pz.MinCost(),
                                  available_models=available_models,
                                  allow_bonded_query=True,
                                  allow_model_selection=False,
                                  allow_code_synth=False,
                                  execution_engine=NoSentinelExecution)
    for record in records:
        print(record.sender, record.subject)


def test_conventional_convert(email_schema):
    """Test whether convert operators"""
    model = pz.Model.GPT_4

    emails = pz.Dataset("enron-eval-tiny", schema=email_schema)
        
    scanOp = MarshalAndScanDataOp(outputSchema=pz.File, dataset_type="dir", shouldProfile=True)
    hardcodedOp = ConvertFileToText(inputSchema=pz.File, outputSchema=pz.TextFile, shouldProfile=True)
    op_class = type('LLMConvert', 
                    (LLMConvertConventional,), 
                    {'model': model, 
                     "prompt_strategy": PromptStrategy.DSPY_COT_QA})
    convertOp = op_class(
        inputSchema=pz.File, 
        outputSchema=email_schema,
        shouldProfile=True)
 
    datasource = DataDirectory().getRegisteredDataset("enron-eval-tiny")
    candidate = DataRecord(schema=pz.File, parent_uuid=None, scan_idx=0)
    candidate.idx = 0
    candidate.get_item_fn = datasource.getItem
    candidate.cardinality = datasource.cardinality
    # run DataSourcePhysicalOp on record

    outputs = []
    records, _ = scanOp(candidate)
    for record in records:
        record, _ = hardcodedOp(record)
        output, _ = convertOp(record[0])
        outputs.extend(output)

    for record in outputs:
        print(record.sender, record.subject)