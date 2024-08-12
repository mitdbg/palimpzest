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
from utils import remove_cache


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
        cardinality="oneToOne",
    )

    with pytest.raises(Exception):
        conv_implementations = pz.ConvertFileToText(
            inputSchema=email_schema,
            outputSchema=email_schema,
            cardinality="oneToOne",
        )

# TODO: uncomment once I understand what is supposed to be happening with
#       ParallelConvertFromCandidateOp and ParallelFilterCandidateOp (I don't
#       have these on my branch; possibly came from another branch)

# def test_convert(email_schema):
#     """Test the physical operators equality sign"""
#     remove_cache()

#     params = {
#         "outputSchema": email_schema,
#         "inputSchema": File,
#         "model": pz.Model.GPT_3_5,
#         "cardinality": "oneToOne",
#     }

#     # simpleConvert = pz.Convert(**params)
#     parallelConvert = pz.ParallelConvertFromCandidateOp(**params, streaming="")
#     monolityhConvert = pz.ConvertOp(**params)

#     assert parallelConvert == parallelConvert
#     assert monolityhConvert == monolityhConvert
#     assert parallelConvert != monolityhConvert

#     print(str(parallelConvert))
#     print(str(monolityhConvert))

#     a = parallelConvert.copy()
#     b = monolityhConvert.copy()
#     assert a == parallelConvert
#     assert b == monolityhConvert
#     assert a != b

# def test_filter(email_schema):
#     """Test the physical operators filter"""
#     remove_cache()

#     params = {
#         "outputSchema": email_schema,
#         "inputSchema": email_schema,
#         "filter": pz.Filter("This is a sample filter"),
#     }

#     # simpleConvert = pz.Convert(**params)
#     parallelFilter = pz.ParallelFilterCandidateOp(**params, streaming="")
#     monoFilter = pz.NonLLMFilter(**params)

#     assert parallelFilter == parallelFilter
#     assert monoFilter == monoFilter
#     assert parallelFilter != monoFilter

#     print(str(parallelFilter))
#     print(str(monoFilter))

#     a = parallelFilter.copy()
#     b = monoFilter.copy()
#     assert a == parallelFilter
#     assert b == monoFilter
#     assert a != b
