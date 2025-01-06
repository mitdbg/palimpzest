import os
import pickle
from threading import Lock

import yaml

from palimpzest import constants
from palimpzest.config import Config
from palimpzest.constants import PZ_DIR
from palimpzest.core.data.datasources import (
    FileSource,
    HTMLFileDirectorySource,
    ImageFileDirectorySource,
    MemorySource,
    PDFFileDirectorySource,
    TextFileDirectorySource,
    UserSource,
    XLSFileDirectorySource,
)


class DataDirectorySingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class CacheService:
    """This class manages the cache for the DataDirectory and other misc PZ components.
    Eventually modify this to be durable and to have expiration policies."""

    def __init__(self):
        self.all_caches = {}

    def get_cached_data(self, cache_name, cache_key):
        return self.all_caches.setdefault(cache_name, {}).get(cache_key, None)

    def put_cached_data(self, cache_name, cache_key, cache_val):
        self.all_caches.setdefault(cache_name, {})[cache_key] = cache_val

    def rm_cached_data(self, cache_name):
        if cache_name in self.all_caches:
            del self.all_caches[cache_name]

    def rm_cache(self):
        self.all_caches = {}


class DataDirectory(metaclass=DataDirectorySingletonMeta):
    """The DataDirectory is a registry of data sources."""

    def __init__(self):
        self._registry = {}
        self._cache = {}
        self._tempCache = {}
        self.cacheService = CacheService()

        # set up data directory
        self._dir = PZ_DIR
        current_config_path = os.path.join(self._dir, "current_config.yaml")
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            os.makedirs(self._dir + "/data/registered")
            os.makedirs(self._dir + "/data/cache")
            with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
                pickle.dump(self._registry, f)

            # create default config
            default_config = Config("default")
            default_config.set_current_config()

        # read current config (and dict. of configs) from disk
        self._current_config = None
        if os.path.exists(current_config_path):
            with open(current_config_path) as f:
                current_config_dict = yaml.safe_load(f)
                self._current_config = Config(current_config_dict["current_config_name"])

        # initialize the file cache directory, defaulting to the system's temporary directory "tmp/pz"
        pz_file_cache_dir = self.current_config.get("filecachedir")
        if pz_file_cache_dir and not os.path.exists(pz_file_cache_dir):
            os.makedirs(pz_file_cache_dir)

        # Unpickle the registry of data sources
        if os.path.exists(self._dir + "/data/cache/registry.pkl"):
            with open(self._dir + "/data/cache/registry.pkl", "rb") as f:
                self._registry = pickle.load(f)

        # Iterate through all items in the cache directory, and rebuild the table of entries
        for root, _, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if file.endswith(".cached"):
                    cache_id = file[:-7]
                    self._cache[cache_id] = root + "/" + file

    @property
    def current_config(self):
        if not self._current_config:
            raise Exception("No current config found.")
        return self._current_config

    def get_cache_service(self):
        return self.cacheService

    def get_config(self):
        return self.current_config._load_config()

    def get_file_cache_dir(self):
        return self.current_config.get("filecachedir")

    #
    # These methods handle properly registered data files, meant to be kept over the long haul
    #
    def register_local_directory(self, path, dataset_id):
        """Register a local directory as a data source."""
        self._registry[dataset_id] = ("dir", path)
        with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
            pickle.dump(self._registry, f)

    def register_local_file(self, path, dataset_id):
        """Register a local file as a data source."""
        self._registry[dataset_id] = ("file", path)
        with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
            pickle.dump(self._registry, f)

    def register_dataset(self, vals, dataset_id):
        """Register an in-memory dataset as a data source"""
        self._registry[dataset_id] = ("memory", vals)
        with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
            pickle.dump(self._registry, f)

    def register_user_source(self, src: UserSource, dataset_id: str):
        """Register a user source as a data source."""
        # user sources are always ephemeral
        self._registry[dataset_id] = ("user", src)

    def get_registered_dataset(self, dataset_id):
        """Return a dataset from the registry."""
        if dataset_id not in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")

        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            if all([f.endswith(tuple(constants.IMAGE_EXTENSIONS)) for f in os.listdir(rock)]):
                return ImageFileDirectorySource(rock, dataset_id)
            elif all([f.endswith(tuple(constants.PDF_EXTENSIONS)) for f in os.listdir(rock)]):
                pdfprocessor = self.current_config.get("pdfprocessor")
                if not pdfprocessor:
                    raise Exception("No PDF processor found in the current config.")
                file_cache_dir = self.get_file_cache_dir()
                if not file_cache_dir:
                    raise Exception("No file cache directory found.")
                return PDFFileDirectorySource(
                    path=rock, dataset_id=dataset_id, pdfprocessor=pdfprocessor, file_cache_dir=file_cache_dir
                )
            elif all([f.endswith(tuple(constants.XLS_EXTENSIONS)) for f in os.listdir(rock)]):
                return XLSFileDirectorySource(rock, dataset_id)
            elif all([f.endswith(tuple(constants.HTML_EXTENSIONS)) for f in os.listdir(rock)]):
                return HTMLFileDirectorySource(rock, dataset_id)
            else:
                return TextFileDirectorySource(rock, dataset_id)

        elif entry == "file":
            return FileSource(rock, dataset_id)
        elif entry == "memory":
            return MemorySource(rock, dataset_id)
        elif entry == "user":
            src = rock
            return src
        else:
            raise Exception("Unknown entry type")

    def get_registered_dataset_type(self, dataset_id):
        """Return the type of the given dataset in the registry."""
        if dataset_id not in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")

        entry, _ = self._registry[dataset_id]

        return entry

    def get_cardinality(self, dataset_id):
        """Return the number of records in a dataset."""
        if dataset_id not in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")

        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            # Return the number of files in the directory
            path = rock
            return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Return 1
            return 1
        elif entry == "memory":
            # Return the number of elements in the values list
            return len(rock)
        elif entry == "user":
            return rock.getCardinality()
        else:
            raise Exception("Unknown entry type")

    def list_registered_datasets(self):
        """Return a list of registered datasets."""
        return self._registry.items()

    def rm_registered_dataset(self, dataset_id):
        """Remove a dataset from the registry."""
        del self._registry[dataset_id]
        with open(self._dir + "/data/cache/registry.pkl", "wb") as f:
            pickle.dump(self._registry, f)

    #
    # These methods handle cached results. They are meant to be persisted for performance reasons,
    # but can always be recomputed if necessary.
    #
    def get_cached_result(self, cache_id):
        """Return a cached result."""
        cached_result = None
        if cache_id not in self._cache:
            return cached_result

        with open(self._cache[cache_id], "rb") as f:
            cached_result = pickle.load(f)

        return MemorySource(cached_result, cache_id)

    def clear_cache(self, keep_registry=False):
        """Clear the cache."""
        self._cache = {}
        self._tempCache = {}

        # Delete all files in the cache directory (except registry.pkl if keep_registry=True)
        for root, _, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if os.path.basename(file) != "registry.pkl" or keep_registry is False:
                    os.remove(root + "/" + file)

    def has_cached_answer(self, cache_id):
        """Check if a dataset is in the cache."""
        return cache_id in self._cache

    def open_cache(self, cache_id):
        if cache_id is not None and cache_id not in self._cache and cache_id not in self._tempCache:
            self._tempCache[cache_id] = []
            return True
        return False

    def append_cache(self, cache_id, data):
        self._tempCache[cache_id].append(data)

    def close_cache(self, cache_id):
        """Close the cache."""
        filename = self._dir + "/data/cache/" + cache_id + ".cached"
        try:
            with open(filename, "wb") as f:
                pickle.dump(self._tempCache[cache_id], f)
        except pickle.PicklingError:
            print("Warning: Failed to save cache due to pickling error")
            os.remove(filename)
        del self._tempCache[cache_id]
        self._cache[cache_id] = filename

    def exists(self, dataset_id):
        print("Checking if exists", dataset_id, "in", self._registry)
        return dataset_id in self._registry

    def get_path(self, dataset_id):
        if dataset_id not in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        entry, path = self._registry[dataset_id]
        return path
