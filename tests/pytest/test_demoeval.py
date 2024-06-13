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


def test_enron(enron_eval):
    """Test the enron demo"""
    dataset = enron_eval
    logical = LogicalPlanner()
    physical = PhysicalPlanner()

    plans = logical.generate_plans(dataset)
    lp = plans[0]
    print(lp)
    physicalPlans = physical.generate_plans(lp)
    pp = physicalPlans[0]
    print(pp)

    Execute(dataset, policy=pz.MinCost())
    # execution.execute(pp)