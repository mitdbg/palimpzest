from palimpzest.elements import DataRecord
from .loaders import DirectorySource

class _DataDirectory:
    """The DataDirectory is a registry of data sources."""

    def __init__(self):
        self._registry = {}
        self._cache = {}
        self._tempCache = {}

    def registerLocalDirectory(self, path, uniqName):
        """Register a local directory as a data source."""
        self._registry[uniqName] = DirectorySource(path)

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
        self._cache[cacheId] = self._tempCache[cacheId]
        del self._tempCache[cacheId]

    def getDataset(self, uniqName):
        """Return a dataset from the registry."""
        if uniqName in self._cache:
            return self._cache[uniqName]
        else:
            return self._registry[uniqName]
    
DataDirectory = _DataDirectory()
