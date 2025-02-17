# Session constants
TIMEOUT_SECS = 600
MAX_SESSION_LIFETIME_SECS = 180
MAX_CONNECTION_RETRIES = 2
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 0.5
MAX_RETRY_DELAY = 8.0

# API defaults
BASE_URL = "https://api.together.xyz/v1"

# Download defaults
DOWNLOAD_BLOCK_SIZE = 10 * 1024 * 1024  # 10 MB
DISABLE_TQDM = False

# Messages
MISSING_API_KEY_MESSAGE = """TOGETHER_API_KEY not found.
Please set it as an environment variable or set it as together.api_key
Find your TOGETHER_API_KEY at https://api.together.xyz/settings/api-keys"""

# Minimum number of samples required for fine-tuning file
MIN_SAMPLES = 1

# the number of bytes in a gigabyte, used to convert bytes to GB for readable comparison
NUM_BYTES_IN_GB = 2**30

# maximum number of GB sized files we support finetuning for
MAX_FILE_SIZE_GB = 4.9

# expected columns for Parquet files
PARQUET_EXPECTED_COLUMNS = ["input_ids", "attention_mask", "labels"]
