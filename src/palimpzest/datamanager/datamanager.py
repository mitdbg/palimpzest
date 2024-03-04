from palimpzest.config import Config
from palimpzest.constants import PZ_DIR
from palimpzest.datasources import DirectorySource, FileSource, MemorySource

import os
import pickle
import sys
import yaml

# TODO: I have intentionally only swapped uniqName --> dataset_id in places where
#       I am at least 50% sure that the intended use case of the function is related
#       to working with a Dataset. I still need to think through how to properly
#       re-use computation of Sets by writing/reading them to/from the cache.
#       In cases where I think we may want to use a Set's uid to enable this form of
#       computation re-use, I've left the variable name as `uniqName`

# TODO: rename to DataManager
# TODO: possibly rename to the PZManager, as it also manages the current config
class DataDirectory:
    """The DataDirectory is a registry of data sources."""

    def __init__(self):
        self._registry = {}
        self._cache = {}
        self._tempCache = {}

        # set up data directory
        self._dir = PZ_DIR
        current_config_path = os.path.join(self._dir, "current_config.yaml")
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
            os.makedirs(self._dir + "/data/registered")
            os.makedirs(self._dir + "/data/cache")
            pickle.dump(self._registry, open(self._dir + "/data/cache/registry.pkl", "wb"))

            # create default config
            default_config = Config("default")
            default_config.set_current_config()

        # read current config (and dict. of configs) from disk
        self.current_config = None
        if os.path.exists(current_config_path):
            with open(current_config_path, 'r') as f:
                current_config_dict = yaml.safe_load(f)
                self.current_config = Config(current_config_dict['current_config_name'])

        # initialize the file cache directory, defaulting to the system's temporary directory "tmp/pz"
        pz_file_cache_dir = self.current_config.get("filecachedir")
        if not os.path.exists(pz_file_cache_dir):
            os.makedirs(pz_file_cache_dir)

        # Unpickle the registry of data sources
        if os.path.exists(self._dir + "/data/cache/registry.pkl"):
            self._registry = pickle.load(open(self._dir + "/data/cache/registry.pkl", "rb"))

        # Iterate through all items in the cache directory, and rebuild the table of entries
        for root, _, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if file.endswith(".cached"):
                    uniqname = file[:-7]
                    self._cache[uniqname] = root + "/" + file

    def getConfig(self):
        return self.current_config._load_config()

    def getFileCacheDir(self):
        return self.current_config.get("filecachedir")

    #
    # These methods handle properly registered data files, meant to be kept over the long haul
    #
    def registerLocalDirectory(self, path, dataset_id):
        """Register a local directory as a data source."""
        self._registry[dataset_id] = ("dir", path)
        pickle.dump(self._registry, open(self._dir + "/data/cache/registry.pkl", "wb"))

    def registerLocalFile(self, path, dataset_id):
        """Register a local file as a data source."""
        self._registry[dataset_id] = ("file", path)
        pickle.dump(self._registry, open(self._dir + "/data/cache/registry.pkl", "wb"))

    def registerDataset(self, vals, dataset_id):
        """Register an in-memory dataset as a data source"""
        self._registry[dataset_id] = ("memory", vals)
        pickle.dump(self._registry, open(self._dir + "/data/cache/registry.pkl", "wb"))

    def getRegisteredDataset(self, dataset_id):
        """Return a dataset from the registry."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        
        entry, rock = self._registry[dataset_id]
        if entry == "dir":
            return DirectorySource(rock, dataset_id)
        elif entry == "file":
            return FileSource(rock, dataset_id)
        elif entry == "memory":
            return MemorySource(rock, dataset_id)
        else:
            raise Exception("Unknown entry type")

    def getRegisteredDatasetType(self, dataset_id):
        """Return the type of the given dataset in the registry."""
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")

        entry, _ = self._registry[dataset_id]

        return entry

    def getSize(self, uniqName):
        """Return the size (in bytes) of a dataset."""
        if not uniqName in self._registry:
            raise Exception("Cannot find dataset", uniqName, "in the registry.")
        
        entry, rock = self._registry[uniqName]
        if entry == "dir":
            # Sum the size in bytes of every file in the directory
            path = rock
            return sum([os.path.getsize(os.path.join(path, name)) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Get the size of the file in bytes
            path = rock
            return os.path.getsize(path)
        elif entry == "memory":
            # get the size of the values in bytes
            return sys.getsizeof(rock)
        else:
            raise Exception("Unknown entry type")

    def getCardinality(self, uniqName):
        """Return the number of records in a dataset."""
        if not uniqName in self._registry:
            raise Exception("Cannot find dataset", uniqName, "in the registry.")
        
        entry, rock = self._registry[uniqName]
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
        else:
            raise Exception("Unknown entry type")

    def listRegisteredDatasets(self):
        """Return a list of registered datasets."""
        return self._registry.items()

    def rmRegisteredDataset(self, dataset_id):
        """Remove a dataset from the registry."""
        del self._registry[dataset_id]
        pickle.dump(self._registry, open(self._dir + "/data/cache/registry.pkl", "wb"))

    #
    # These methods handle cached results. They are meant to be persisted for performance reasons,
    # but can always be recomputed if necessary.
    #
    def getCachedResult(self, uniqName):
        """Return a cached result."""
        if not uniqName in self._cache:
            return None

        cachedResult = pickle.load(open(self._cache[uniqName], "rb"))
        def iterateOverCachedResult():
            for x in cachedResult:
                yield x
        return iterateOverCachedResult()

    def clearCache(self, keep_registry=False):
        """Clear the cache."""
        self._cache = {}
        self._tempCache = {}

        # Delete all files in the cache directory (except registry.pkl if keep_registry=True)
        for root, dirs, files in os.walk(self._dir + "/data/cache"):
            for file in files:
                if os.path.basename(file) != "registry.pkl" or keep_registry is False:
                    os.remove(root + "/" + file)

    def hasCachedAnswer(self, uniqName):
        """Check if a dataset is in the cache."""
        return uniqName in self._cache

    def openCache(self, cacheId):
        if not cacheId is None and not cacheId in self._cache and not cacheId in self._tempCache:
            self._tempCache[cacheId] = []
            return True
        return False

    def appendCache(self, cacheId, data):
        self._tempCache[cacheId].append(data)

    def closeCache(self, cacheId):
        """Close the cache."""
        filename = self._dir + "/data/cache/" + cacheId + ".cached"
        pickle.dump(self._tempCache[cacheId], open(filename, "wb"))
        del self._tempCache[cacheId]
        self._cache[cacheId] = filename

    def exists(self, dataset_id):
        print("Checking if exists", dataset_id, "in", self._registry)
        return dataset_id in self._registry

    def getPath(self, dataset_id):
        if not dataset_id in self._registry:
            raise Exception("Cannot find dataset", dataset_id, "in the registry.")
        entry, path = self._registry[dataset_id]
        return path
