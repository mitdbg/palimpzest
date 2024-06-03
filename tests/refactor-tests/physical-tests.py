""" This script contains tests for the refactoring of the physical operators
"""

import os
import sys
import time
import pdb

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")
import context

import unittest
import palimpzest as pz
from palimpzest.planner import SimplePlanner

from utils import remove_cache, buildNestedStr


class Email(pz.TextFile):
    """Represents an email, which in practice is usually from a text file"""

    sender = pz.Field(desc="The email address of the sender", required=True)
    subject = pz.Field(desc="The subject of the email", required=True)


def EnronEval():
    emails = pz.Dataset("enron-eval", schema=Email)
    emails = emails.filterByStr(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filterByStr(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails


def EnronTiny():
    emails = pz.Dataset("enron-tiny", schema=Email)
    emails = emails.filterByStr("The email is about someone taking a vacation")
    return emails


# NOTE: the following weird iteration over physical plans by idx is intentional and necessary
#       at the moment in order for stats collection to work properly. For some yet-to-be-discovered
#       reason, `createPhysicalPlanCandidates` is creating physical plans which share the same
#       copy of some operators. This means that if we naively iterate over the plans and execute them
#       some plans' profilers will count 2x (or 3x or 4x etc.) the number of records processed,
#       dollars spent, time spent, etc. This workaround recreates the physical plans on each
#       iteration to ensure that they are new.


class TestPhysicalOperators(unittest.TestCase):

    def test_no_print(self, limit=1):
        """Disable the print statement from each record"""
        remove_cache()

        dataset = EnronTiny()
        planner = SimplePlanner()
        #
        logical_plan = planner.plan_logical(dataset)
        physical = planner.plan_physical(logical_plan, max=limit, shouldProfile=True)

        logicalTree = dataset.getLogicalTree()

        candidatePlans = logicalTree.createPhysicalPlanCandidates(
            max=limit, shouldProfile=True
        )

    def test_physOp(
        self,
    ):
        """Test the physical operators equality sign"""
        remove_cache()

        params = {
            "outputSchema": Email,
            "source": pz.CacheScanDataOp(outputSchema=Email, cacheIdentifier=""),
            "model": pz.Model.GPT_3_5,
            "cardinality": "oneToOne",
        }

        # simpleInduce = pz.Induce(**params)
        parallelInduce = pz.ParallelInduceFromCandidateOp(**params, streaming="")
        monolityhInduce = pz.InduceFromCandidateOp(**params)

        assert parallelInduce == parallelInduce
        assert monolityhInduce == monolityhInduce
        assert parallelInduce != monolityhInduce

        print(str(parallelInduce))
        print(str(monolityhInduce))

        a = parallelInduce.copy()
        b = monolityhInduce.copy()
        assert a == parallelInduce
        assert b == monolityhInduce
        assert a != b

    def test_duplicate_plans(self):
        "We want this test to understand why some physical plans share the same copy of some operators"
        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        pass


if __name__ == "__main__":
    unittest.main()
