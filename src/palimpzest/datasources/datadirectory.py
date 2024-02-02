from palimpzest.elements import DataRecord
from .loaders import DirectorySource, FileSource

import os
import pickle

class _DataDirectory:
    """The DataDirectory is a registry of data sources."""

    def __init__(self, configDir, create=False):
        self._registry = {}
        self._cache = {}
        self._tempCache = {}

        self._configDir = configDir
        if create:
            if not os.path.exists(configDir):
                os.makedirs(configDir)
                os.makedirs(configDir + "/registered")
                os.makedirs(configDir + "/cache")
                pickle.dump(self._registry, open(configDir + "/cache/registry.pkl", "wb"))

        # Unpickle the registry of data sources
        if os.path.exists(configDir + "/cache/registry.pkl"):
            self._registry = pickle.load(open(configDir + "/cache/registry.pkl", "rb"))

        # Iterate through all items in the cache directory, and rebuild the table of entries
        for root, dirs, files in os.walk(configDir + "/cache"):
            for file in files:
                if file.endswith(".cached"):
                    uniqname = file[:-7]
                    self._cache[uniqname] = root + "/" + file

    #
    # These methods handle properly registered data files, meant to be kept over the long haul
    #
    def registerLocalDirectory(self, path, uniqName):
        """Register a local directory as a data source."""
        self._registry[uniqName] = ("dir", path)
        pickle.dump(self._registry, open(self._configDir + "/cache/registry.pkl", "wb"))

    def registerLocalFile(self, path, uniqName):
        """Register a local file as a data source."""
        self._registry[uniqName] = ("file", path)
        pickle.dump(self._registry, open(self._configDir + "/cache/registry.pkl", "wb"))

    def getRegisteredDataset(self, uniqName):
        """Return a dataset from the registry."""
        if not uniqName in self._registry:
            raise Exception("Cannot find dataset", uniqName, "in the registry.")
        
        entry, path = self._registry[uniqName]
        if entry == "dir":
            return DirectorySource(path)
        elif entry == "file":
            # THIS IS NOT RETURNING A GOOD ITERATOR SOMEHOW!!!!!
            return FileSource(path)
        else:
            raise Exception("Unknown entry type")

    def getSize(self, uniqName):
        """Return the size (in bytes) of a dataset."""
        if not uniqName in self._registry:
            raise Exception("Cannot find dataset", uniqName, "in the registry.")
        
        entry, path = self._registry[uniqName]
        if entry == "dir":
            # Sum the length in bytes of every file in the directory
            return sum([os.path.getsize(os.path.join(path, name)) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Get the length of the file
            return os.path.getsize(path)
        else:
            raise Exception("Unknown entry type")

    def getCardinality(self, uniqName):
        """Return the number of records in a dataset."""
        if not uniqName in self._registry:
            raise Exception("Cannot find dataset", uniqName, "in the registry.")
        
        entry, path = self._registry[uniqName]
        if entry == "dir":
            # Return the number of files in the directory
            return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
        elif entry == "file":
            # Return 1
            return 1
        else:
            raise Exception("Unknown entry type")

    def listRegisteredDatasets(self):
        """Return a list of registered datasets."""
        return self._registry.items()
    
    def rmRegisteredDataset(self, uniqName):
        """Remove a dataset from the registry."""
        del self._registry[uniqName]
        pickle.dump(self._registry, open(self._configDir + "/cache/registry.pkl", "wb"))
    
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
    
    def clearCache(self):
        """Clear the cache."""
        self._cache = {}
        self._tempCache = {}

        # Delete all files in the cache directory
        for root, dirs, files in os.walk(self._configDir + "/cache"):
            for file in files:
                os.remove(root + "/" + file)

    def hasCachedAnswer(self, uniqName):
        """Check if a dataset is in the cache."""
        return uniqName in self._cache

    def openCache(self, cacheId):
        if not cacheId in self._cache and not cacheId in self._tempCache:
            self._tempCache[cacheId] = []
            return True
        return False

    def appendCache(self, cacheId, data):
        self._tempCache[cacheId].append(data)

    def closeCache(self, cacheId):
        """Close the cache."""
        filename = self._configDir + "/cache/" + cacheId + ".cached"
        pickle.dump(self._tempCache[cacheId], open(filename, "wb"))
        del self._tempCache[cacheId]
        self._cache[cacheId] = filename

_DataDirectoryMember = None
def initDataDirectory(initDir, create=False):
    """Initialize the DataDirectory with a directory."""
    global _DataDirectoryMember
    if not _DataDirectoryMember is None:
        raise Exception("DataDirectory already initialized")
    else:
        _DataDirectoryMember = _DataDirectory(initDir, create=create)
        return _DataDirectoryMember

def DataDirectory():
    return _DataDirectoryMember