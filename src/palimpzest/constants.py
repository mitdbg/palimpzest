### This file contains constants used by Palimpzest ###
import os

# Palimpzest root directory
PZ_DIR = os.path.join(os.path.expanduser("~"), ".palimpzest")

# Assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# Assume 10s per record for local LLM object conversion
STD_LLM_CONVERSION_TIME_PER_RECORD = 20
PARALLEL_LLM_CONVERSION_TIME_OVERALL = 2.0 * STD_LLM_CONVERSION_TIME_PER_RECORD

# Assume 0.06 per 1M tokens, and about 4K tokens per request (way wrong)
STD_LLM_CONVERSION_COST_PER_RECORD = 0.06 * (4000 / 1000000)
PARALLEL_LLM_CONVERSION_COST_PER_RECORD = STD_LLM_CONVERSION_COST_PER_RECORD

# Assume filter operations are twice as fast as conversions
STD_LLM_FILTER_TIME_PER_RECORD = STD_LLM_CONVERSION_TIME_PER_RECORD / 2
PARALLEL_LLM_FILTER_TIME_OVERALL = PARALLEL_LLM_CONVERSION_TIME_OVERALL / 2

STD_LLM_FILTER_COST_PER_RECORD = STD_LLM_CONVERSION_COST_PER_RECORD / 2
PARALLEL_LLM_FILTER_COST_PER_RECORD = PARALLEL_LLM_CONVERSION_COST_PER_RECORD / 2

# Whether or not to log LLM outputs
LOG_LLM_OUTPUT = False
