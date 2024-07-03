""" This script contains tests for the refactoring of the physical operators
"""

import os
import sys
import time
import pdb

from palimpzest.corelib import File, TextFile

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")
import context

import pytest
import palimpzest as pz
from palimpzest.planner import LogicalPlanner, PhysicalPlanner
from palimpzest.operators import ConvertOp, ConvertFileToText
from utils import remove_cache, buildNestedStr


# NOTE: the following weird iteration over physical plans by idx is intentional and necessary
#       at the moment in order for stats collection to work properly. For some yet-to-be-discovered
#       reason, `createPhysicalPlanCandidates` is creating physical plans which share the same
#       copy of some operators. This means that if we naively iterate over the plans and execute them
#       some plans' profilers will count 2x (or 3x or 4x etc.) the number of records processed,
#       dollars spent, time spent, etc. This workaround recreates the physical plans on each
#       iteration to ensure that they are new.

def test_implement(enron_eval):

    l_op = pz.BaseScan
    p_op = pz.MarshalAndScanDataOp

    assert p_op.implements(l_op)

def test_physical_planner():
    """Test the physical planner"""
    physical = PhysicalPlanner()

def test_class_attributes(email_schema):
    generic_convert = pz.ConvertOp
    conv_file_text = pz.ConvertFileToText

    print("Input schema of ConvertOp: ", generic_convert.inputSchema)
    print("Output schema of ConvertOp: ", generic_convert.outputSchema)
    print("Input schema of ConvertFileToText: ", conv_file_text.inputSchema)
    print("Output schema of ConvertFileToText: ", conv_file_text.outputSchema)

    assert generic_convert.inputSchema != conv_file_text.inputSchema
    assert generic_convert.outputSchema != conv_file_text.outputSchema

    conv_implementations = pz.ConvertFileToText(
        inputSchema=File,
        outputSchema=TextFile,
        model=pz.Model.GPT_3_5,
        cardinality="oneToOne",
    )

    with pytest.raises(Exception):
        conv_implementations = pz.ConvertFileToText(
            inputSchema=email_schema,
            outputSchema=email_schema,
            model=pz.Model.GPT_3_5,
            cardinality="oneToOne",
        )

def test_logical(enron_eval):
    """Test whether logical plans work"""
    remove_cache()

    dataset = enron_eval
    logical = LogicalPlanner()
    logical.generate_plans(dataset)
    logical_plan = next(logical)
    print(logical_plan)

def test_convert(email_schema):
    """Test the physical operators equality sign"""
    remove_cache()

    params = {
        "outputSchema": email_schema,
        "inputSchema": File,
        "model": pz.Model.GPT_3_5,
        "cardinality": "oneToOne",
    }

    # simpleConvert = pz.Convert(**params)
    parallelConvert = pz.ParallelConvertFromCandidateOp(**params, streaming="")
    monolityhConvert = pz.ConvertOp(**params)

    assert parallelConvert == parallelConvert
    assert monolityhConvert == monolityhConvert
    assert parallelConvert != monolityhConvert

    print(str(parallelConvert))
    print(str(monolityhConvert))

    a = parallelConvert.copy()
    b = monolityhConvert.copy()
    assert a == parallelConvert
    assert b == monolityhConvert
    assert a != b

def test_filter(email_schema):
    """Test the physical operators filter"""
    remove_cache()

    params = {
        "outputSchema": email_schema,
        "inputSchema": email_schema,
        "filter": pz.Filter("This is a sample filter"),
    }

    # simpleConvert = pz.Convert(**params)
    parallelFilter = pz.ParallelFilterCandidateOp(**params, streaming="")
    monoFilter = pz.NonLLMFilter(**params)

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

def test_duplicate_plans():
    "We want this test to understand why some physical plans share the same copy of some operators"
    # TODO: for now, re-create candidate plans until we debug duplicate profiler issue
    pass