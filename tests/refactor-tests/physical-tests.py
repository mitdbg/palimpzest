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

    def test_no_keys(self, limit=1):
        """Buggy behavior: if OPEN API keys are not set but only GOOGLE, the physical plan generation fails instead of skipping the model
        For some reason, the generator is still passed OPENAI models even though the keys are not set
        That leads to a crash.
        Steps to reproduce this bug: delete the keys from the environment and run the enron emails test
        """
        del os.environ["OPENAI_API_KEY"]
        del os.environ["TOGETHER_API_KEY"]
        del os.environ["GOOGLE_API_KEY"]
        print(os.environ["GOOGLE_API_KEY"])
        remove_cache()

        dataset = EnronTiny()
        logicalTree = dataset.getLogicalTree()

        candidatePlans = logicalTree.createPhysicalPlanCandidates(
            max=limit, shouldProfile=True
        )
        # TODO: candidatePlans should not return a TUPLE
        # A PLAN SHOULD BE A CLASS
        for idx, plan_tuple in enumerate(candidatePlans):
            plan = plan_tuple[3]
            print("----------------------")
            print(f"Plan: {buildNestedStr(plan.dumpPhysicalTree())}")
            print("---")

            # TODO do not print the record every single time
            records = [r for r in plan]

            assert len(records) > 0

    def test_duplicate_plans(self):
        "We want this test to understand why some physical plans share the same copy of some operators"
        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        pass


if __name__ == "__main__":
    unittest.main()
