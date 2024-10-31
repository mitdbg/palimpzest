"""This script contains tests for the refactoring of the physical operators"""

import os
import sys

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


# TODO: uncomment once I understand what is supposed to be happening with
#       ParallelConvertFromCandidateOp and ParallelFilterCandidateOp (I don't
#       have these on my branch; possibly came from another branch)

# def test_convert(email_schema):
#     """Test the physical operators equality sign"""
#     remove_cache()

#     params = {
#         "output_schema": email_schema,
#         "input_schema": File,
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
#         "output_schema": email_schema,
#         "input_schema": email_schema,
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
