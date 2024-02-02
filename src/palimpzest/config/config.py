from palimpzest.datasources import DataDirectory, initDataDirectory

import yaml
import os
import sys

class Config:
    def __init__(self, path, create=False):
        self.path = path
        self.datadirectorypath = os.path.join(path, "data")
        self.configfilepath = os.path.join(path, "config.yaml")

        if create:
            if not os.path.exists(path):
                os.makedirs(path)
            initDataDirectory(self.datadirectorypath, create=True)
            if not os.path.exists(self.configfilepath):
                with open(self.configfilepath, "w") as file:
                    self.config = {"data_directory": self.datadirectorypath}
                    self.save_config()
        else:
            if not os.path.exists(self.path):
                raise Exception("No configuration information available at " + self.path)
            self.config = self.load_config()
            initDataDirectory(self.config["data_directory"])

    def load_config(self):
        """Load YAML configuration from the specified path."""
        try:
            with open(self.configfilepath, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            sys.exit(1)

    def save_config(self):
        """Save the configuration to the specified path."""
        try:
            with open(self.configfilepath, 'w') as file:
                yaml.dump(self.config, file)
        except Exception as e:
            print(f"Error saving configuration file: {e}")
            sys.exit(1)
