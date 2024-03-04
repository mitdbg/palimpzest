### This file contains constants used by Palimpzest ###
from enum import Enum

import os

# ENUMS
class Model(Enum):
    LLAMA2 = "meta-llama/Llama-2-7b-hf"
    MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    GPT_3_5 = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4-0125-preview"

class PromptStrategy(Enum):
    ZERO_SHOT = "zero-shot"
    FEW_SHOT = "few-shot"
    DSPY_COT = "dspy-chain-of-thought"
    DSPY_BOOL = "dspy-bool"

# Palimpzest root directory
PZ_DIR = os.path.join(os.path.expanduser("~"), ".palimpzest")

# Assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# Assume 30 GB/sec for sequential access of memory
MEMORY_SCAN_TIME_PER_KB = 1 / (float(30) * 1024 * 1024)

# Assume number of output tokens is 0.25x the number of input tokens
OUTPUT_TOKENS_MULTIPLE = 0.25

# Guesstimate of the fraction of each input element which is fed into the LLM's as context 
ELEMENT_FRAC_IN_CONTEXT = 1.0

# Rough conversion from # of bytes --> # of tokens; assumes 1 token ~= 4 chars and 1 char == 1 byte
BYTES_TO_TOKENS = 0.25

# TODO: make this much more of a science than a terrible guess
# Using DSPy as a prompt strategy can lead to increases in cost and time due to the
# fact that DSPy -- when provided with a validation metric -- makes some additional
# API calls to find the optimal prompt(s). For now we will just apply a horrendously
# imprecise and innacurate multiple to our cost and time estimates for generation with
# the DSPy prompt strategy. In the near future we may want to make this much more exact.
DSPY_COST_INFLATION = 2.0
DSPY_TIME_INFLATION = 2.0

# Similar to the DSPY_*_INFLATION variables, we make a crude estimate of the additional
# input token size due to using a few-shot prompt strategy, which can mildly increase the
# size of the input.
FEW_SHOT_PROMPT_INFLATION = 1.25

# A crude estimate for filter selectivity
EST_FILTER_SELECTIVITY = 0.5

# Whether or not to log LLM outputs
LOG_LLM_OUTPUT = False

#### MODEL PERFORMANCE & COST METRICS ####
# I've looked across models and grouped knowledge into commonly used categories:
# - Agg. Benchmark (we only use MMLU for this)
# - Commonsense Reasoning
# - World Knowledge
# - Reading Comprehension
# - Code
# - Math
#
# We don't have global overlap on the World Knowledge and/or Reading Comprehension
# datasets. Thus, we include these categories results where we have them, but they
# are completely omitted for now.
#
# Within each category only certain models have overlapping results on the same
# individual datasets; in order to have consistent evaluations I have computed
# the average result for each category using only the shared sets of datasets within
# that category. All datasets for which we have results will be shown but commented
# with ###; datasets which are used in our category averages will have a ^.
#
# Cost is presented in terms of USD / token for input tokens and USD / token for
# generated tokens.
# 
# Time is presented in seconds per output token. I grabbed some semi-recent estimates
# from the internet for this quick POC, but we can and should do more to model these
# values more precisely:
# - Llama2 7B: https://blog.truefoundry.com/llama-2-benchmarks/
# - Mixtral 7B: https://artificialanalysis.ai/models/mixtral-8x7b-instruct/hosts
# - GPT 3.5/4: https://community.openai.com/t/gpt-3-5-and-gpt-4-api-response-time-measurements-fyi/237394/16 
LLAMA2_7B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 1 / 1E10,  # for now, let's have a de minimis cost for Llama2
    "usd_per_output_token": 1 / 1E10,
    ##### Time #####
    "seconds_per_output_token": 0.005, # Assuming an A10 (i.e. G5 on EC2); see link for A100 est.
    ##### Agg. Benchmark #####
    "MMLU": 45.3,
    ### "BBH": 32.6,
    ### "AGI Eval": 29.3,
    ##### Commonsense Reasoning #####
    "reasoning": 73.9,
    ### "HellaSwag": 77.2,^
    ### "WinoGrande": 69.2,^
    ### "PIQA": 78.8,
    ### "SIQA": 48.3,
    ### "Arc-e": 75.2,^
    ### "Arc-c": 45.9,
    ##### World Knowledge #####
    ### "NaturalQuestions": 25.7, # 5-shot
    ### "TriviaQA": 72.1,         # 5-shot
    ##### Reading Comprehension #####
    ### "SQuAD": 62.8, # 5-shot
    ### "QuAC": 39.7,  # 1-shot
    ### "BoolQ": 77.4,
    ##### Code #####
    "code": 12.8,
    ### "HumanEval": 12.8,^ # pass@1
    ### "MBPP": 20.8,       # pass@1
    ##### Math #####
    "math": 14.6,
    ### "MATH": 2.5,
    ### "GSM8K": 14.6,^
}
MIXTRAL_8X_7B_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.7 / 1E6,
    "usd_per_output_token": 0.7 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.009,
    ##### Agg. Benchmark #####
    "MMLU": 70.6,
    ##### Commonsense Reasoning #####
    "reasoning": 81.6,
    ### "HellaSwag": 84.4,^  # 10-shot
    ### "WinoGrande": 77.2,^ # 5-shot
    ### "PIQA": 83.6,
    ### "Arc-e": 83.1,^      # 25-shot
    ### "Arc-c": 59.7,
    ##### World Knowledge #####
    ### "NaturalQuestions": 30.6,
    ### "TriviaQA": 71.5,
    ##### Reading Comprehension #####
    ##### Code #####
    "code": 40.2,
    ### "HumanEval": 40.2,
    ### "MBPP": 60.7,      # pass@1
    ##### Math #####
    "math": 74.4,
    ### "MATH": 28.4,
    ### "GSM8K": 74.4, # 5-shot
}
# NOTE: seconds_per_output_token is based on `gpt-3.5-turbo-1106`
GPT_3_5_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 0.5 / 1E6,
    "usd_per_output_token": 1.5 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.0065,
    ##### Agg. Benchmark #####
    "MMLU": 70.0, # 5-shot
    ##### Commonsense Reasoning #####
    "reasoning": 84.1,
    ### "HellaSwag": 85.5,^  # 10-shot
    ### "WinoGrande": 81.6,^ # 5-shot
    ### "Arc-e": 85.2,^      # 25-shot
    ##### World Knowledge #####
    ##### Reading Comprehension #####
    ### "DROP": 64.1, # 3-shot
    ##### Code #####
    "code": 48.1,
    ### "HumanEval": 48.1,^ # 0-shot    
    ##### Math #####
    "math": 57.1,
    ### "GSM8K": 57.1,^     # 5-shot
}
# NOTE: the seconds_per_output_token was computed based on a slightly different model ('gpt-4-1106-preview')
#       and the benchmark statistics were computed based on the GPT-4 Technical Report; these might be
#       slightly innacurate compared to the real numbers for gpt-4-0125-preview, but we'll use them until
#       we have something better. (The cost metrics are accurate).
GPT_4_MODEL_CARD = {
    ##### Cost in USD #####
    "usd_per_input_token": 10 / 1E6,
    "usd_per_output_token": 30 / 1E6,
    ##### Time #####
    "seconds_per_output_token": 0.018,
    ##### Agg. Benchmark #####
    "MMLU": 86.4, # 5-shot
    ##### Commonsense Reasoning #####
    "reasoning": 93.0,
    ### "HellaSwag": 95.3,^  # 10-shot
    ### "WinoGrande": 87.5,^ # 5-shot
    ### "Arc-e": 96.3,^      # 25-shot
    ##### World Knowledge #####
    ##### Reading Comprehension #####
    ### "DROP": 80.9, # 3-shot
    ##### Code #####
    "code": 67.0,
    ### "HumanEval": 67.0,^ # 0-shot
    ##### Math #####
    "math": 92.0,
    ### "GSM8K": 92.0,^     # 5-shot
}

MODEL_CARDS = {
    Model.LLAMA2.value: LLAMA2_7B_MODEL_CARD,
    Model.MIXTRAL.value: MIXTRAL_8X_7B_MODEL_CARD,
    Model.GPT_3_5.value: GPT_3_5_MODEL_CARD,
    Model.GPT_4.value: GPT_4_MODEL_CARD,
}