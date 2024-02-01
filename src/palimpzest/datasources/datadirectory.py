from palimpzest.elements import DataRecord
from .loaders import DirectorySource

class _DataDirectory:
    """The DataDirectory is a registry of data sources."""

    def __init__(self):
        self._registry = {}

    def registerLocalDirectory(self, path, uniqName):
        """Register a local directory as a data source."""
        self._registry[uniqName] = DirectorySource(path)

    def getDataset(self, uniqName):
        """Return a dataset from the registry."""
        return self._registry[uniqName]
    
DataDirectory = _DataDirectory()
