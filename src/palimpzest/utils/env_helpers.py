import os
import sys


def load_env():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # read the env file
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                key, value = line.strip().split("=")
                os.environ[key] = value
