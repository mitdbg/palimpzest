""" This script contains tests for the refactoring of the physical operators
"""

import os
import sys
import time
import pdb

from palimpzest.elements.elements import File, TextFile

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

    def test_physical_planner(self):
        """Test the physical planner"""
        physical = PhysicalPlanner()

    def test_class_attributes(self):
        generic_induce = pz.ConvertOp
        conv_file_text = pz.ConvertFileToText

        print("Input schema of ConvertOp: ", generic_induce.inputSchema)
        print("Output schema of ConvertOp: ", generic_induce.outputSchema)
        print("Input schema of ConvertFileToText: ", conv_file_text.inputSchema)
        print("Output schema of ConvertFileToText: ", conv_file_text.outputSchema)

        assert generic_induce.inputSchema != conv_file_text.inputSchema
        assert generic_induce.outputSchema != conv_file_text.outputSchema

        conv_implementations = pz.ConvertFileToText(
            inputSchema=File,
            outputSchema=TextFile,
            model=pz.Model.GPT_3_5,
            cardinality="oneToOne",
        )

        conv_implementations = pz.ConvertFileToText(
            inputSchema=Email,
            outputSchema=Email,
            model=pz.Model.GPT_3_5,
            cardinality="oneToOne",
        )

    def test_end_to_end(self):
        """Test the end-to-end physical planner"""
        remove_cache()

        dataset = EnronTiny()
        # Todo add execution class

    def test_logical(self, limit=1):
        """Test whether logical plans work"""
        remove_cache()

        dataset = EnronTiny()
        logical = LogicalPlanner()
        logical.generate_plans(dataset)
        logical_plan = next(logical)
        print(logical_plan)

    def test_induce(
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

    def test_filter(
        self,
    ):
        """Test the physical operators filter"""
        remove_cache()

        params = {
            "outputSchema": Email,
            "source": pz.CacheScanDataOp(outputSchema=Email, cacheIdentifier=""),
            "model": pz.Model.GPT_3_5,
            "filter": pz.Filter("This is a sample filter"),
        }

        # simpleInduce = pz.Induce(**params)
        parallelFilter = pz.ParallelFilterCandidateOp(**params, streaming="")
        monoFilter = pz.FilterCandidateOp(**params)

        assert parallelFilter == parallelFilter
        assert monoFilter == monoFilter
        assert parallelFilter != monoFilter

        print(str(parallelFilter))
        print(str(monoFilter))

        a = parallelFilter.copy()
        b = monoFilter.copy()
        assert a == parallelFilter
        assert b == monoFilter
        assert a != b

    def test_duplicate_plans(self):
        "We want this test to understand why some physical plans share the same copy of some operators"
        # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
        pass


if __name__ == "__main__":
    unittest.main()
