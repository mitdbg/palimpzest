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


def test_strategy(enron_eval):
    """Test whether strategy creation works"""
    available_models = [pz.Model.GPT_4, pz.Model.GPT_3_5]

    strategy = ModelSelectionFilterStrategy(available_models=available_models)
    print(strategy)

    records, plan, stats= Execute(enron_eval, 
                                  policy=pz.MinCost(),
                                  available_models=available_models,
                                  useStrategies=True,
                                  allow_model_selection=False,
                                  allow_code_synth=False)
