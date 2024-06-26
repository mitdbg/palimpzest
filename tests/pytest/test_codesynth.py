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
from palimpzest.execution.nosentinel_execution import NoSentinelExecution


def test_codesynth(email_schema):
    """Test whether codesynth strategy creation works"""
    available_models = [pz.Model.GPT_4]

    emails = pz.Dataset("enron-eval-tiny", schema=email_schema)
    records, plan, stats= Execute(emails, 
                                  policy=pz.MinCost(),
                                  available_models=available_models,
                                  useStrategies=True,
                                  allow_bonded_query=False,
                                  allow_model_selection=False,
                                  allow_code_synth=True,
                                  execution_engine=NoSentinelExecution)
    
    for record in records:
        print(record.sender, record.subject)
