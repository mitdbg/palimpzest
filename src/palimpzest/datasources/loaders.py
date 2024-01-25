from palimpzest.elements import *

class DataSource():
    """The base class for all data sources"""
    def __init__(self, basicElement, desc=None):
        self.basicElement = basicElement
        self.desc = desc

    def __repr__(self):
        return f"{self.__class__.__name__}(basicElement={self.basicElement}, desc={self.desc})"
    
    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__

class DirectorySource(DataSource):
    """DirectorySource returns multiple File objects from a real-world source (a directory on disk)"""
    def __init__(self, path, desc=None):
        self.path = path
        self.basicElement = File(desc=f"A file loaded from {path}")
        super().__init__(self.basicElement, desc=desc)

    def __iter__(self):
        def filteredIterator():
            for x in os.walk(self.path):
                # Somehow we populate the File data????
                yield self.basicElement.DataRecord(self.basicElement, path=x, bytes=open(x, "rb").read())
        return filteredIterator()

#
# Other subclasses of DataSource could grab data from a database, a blob store, etc.
# The basicElement returned might not be a File, but instead a Record or Image or similar.
#