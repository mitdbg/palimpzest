""" This testing class is an integration test suite. 
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import unittest
import os
import sys
import time
import pdb

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")
import context

import unittest
import palimpzest as pz
from palimpzest.planner import LogicalPlanner, PhysicalPlanner
from palimpzest.operators import ConvertOp, ConvertFileToText
from utils import remove_cache, buildNestedStr


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


def EnronEvalTiny():
    emails = pz.Dataset("enron-eval-tiny", schema=Email)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails


def EnronTiny():
    emails = pz.Dataset("enron-tiny", schema=Email)
    emails = emails.filter("The email is about someone taking a vacation")
    return emails


class TestDemo(unittest.TestCase):

    def test_enron(self):
        """Test the enron demo"""
        dataset = EnronEvalTiny()
        logical = LogicalPlanner()
        physical = PhysicalPlanner()

        plans = logical.generate_plans(dataset)
        lp = plans[0]
        print(lp)
        # physicalPlans = physical.generate_plans(lp)
        # pp = physicalPlans[0]


if __name__ == "__main__":
    unittest.main()
