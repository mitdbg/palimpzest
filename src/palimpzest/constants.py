### This file contains constants used by Palimpzest ###
import os

# Palimpzest root directory
PZ_DIR = os.path.join(os.path.expanduser("~"), ".palimpzest")

# Assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# Assume 10s per record for local LLM object conversion
LOCAL_LLM_CONVERSION_TIME_PER_RECORD = 10

# Assume 5s per record for local LLM boolean filter
LOCAL_LLM_FILTER_TIME_PER_RECORD = 5

# Whether or not to log LLM outputs
LOG_LLM_OUTPUT = False
