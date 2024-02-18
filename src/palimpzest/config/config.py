import tempfile

import yaml
import os
import sys

# DEFINITIONS
PZ_DIR = os.path.join(os.path.expanduser("~"), ".palimpzest")


class Config:
    def __init__(self, name: str = "default"):
        self.configfilepath = os.path.join(PZ_DIR, f"config_{name}.yaml")
        if not os.path.exists(PZ_DIR):
            raise Exception(f"Target config directory does not exist at {PZ_DIR} :: Something is wrong with the installation.")

        if not os.path.exists(self.configfilepath):
            # Get the system's temporary directory
            temp_dir = tempfile.gettempdir()
            pz_file_cache_dir = os.path.join(temp_dir, "pz")
            self.config = {"name": name, "llmservice": "openai", "parallel": "False", "filecachedir": pz_file_cache_dir}
            self._save_config()

        self.config = self._load_config()
        if not "filecachedir" in self.config:
            # Get the system's temporary directory
            temp_dir = tempfile.gettempdir()
            pz_file_cache_dir = os.path.join(temp_dir, "pz")
            self.config["filecachedir"] = pz_file_cache_dir
            self._save_config()

    def get(self, key):
        return self.config[key]

    def set(self, key, value):
        self.config[key] = value
        self._save_config()

    def _load_config(self):
        """Load YAML configuration from the specified path."""
        try:
            with open(self.configfilepath, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)

    def _save_config(self):
        """Save the configuration to the specified path."""
        try:
            with open(self.configfilepath, 'w') as file:
                yaml.dump(self.config, file)
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            sys.exit(1)
