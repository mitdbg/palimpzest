import tempfile

import yaml
import os
import sys

class Config:
    def __init__(self, path):
        self.configfilepath = os.path.join(path, "config.yaml")
        if not os.path.exists(path):
            raise Exception("Target config directory does not exist at", path, ". Something is wrong with the installation.")

        if not os.path.exists(self.configfilepath):
            # Get the system's temporary directory
            temp_dir = tempfile.gettempdir()
            pz_file_cache_dir = os.path.join(temp_dir, "pz")
            with open(self.configfilepath, "w") as file:
                self.config = {"llmservice": "openai", "parallel": "False", "filecachedir": pz_file_cache_dir}
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
