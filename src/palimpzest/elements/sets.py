from .elements import *
from .operators import *
from .filters import *

#####################################################
#####################################################
class Set:
    """A Set is set of Elements. It can be iterated over."""
    def __init__(self, basicElt, input=None, desc=None, filters=[]):
        self._desc = desc
        self._basicElt = basicElt
        self._input = input
        self._filters = filters

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self._filters])
        return f"{self.__class__.__name__}(basicElt={self._basicElt}, desc={self._desc}, filters={filterStr})"

    def schema(self):
        """The Set's basic element"""
        return self._basicElt

    def addFilter(self, f: Filter):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        return Set(self._basicElt, input=self, desc="Apply filter(s)", filters=[f])

    def addFilterStr(self, filterCondition: str, targetFn=None):
        """Add a filter to the Set. This filter will possibly restrict the items that are returned later."""
        if targetFn is None:
            targetFn = lambda x: x

        f = Filter(filterCondition, transformingFn=targetFn)
        return self.addFilter(f)
    
    def dumpSyntacticTree(self):
        """Return the syntactic tree of this Set."""
        return (self, None if self._input is None else self._input.dumpSyntacticTree())

    def getLogicalTree(self):
        """Return the logical tree of operators on Sets."""
        if self._input is None:
            return BaseScan(self._basicElt)
        else:
            return FilteredScan(self._basicElt, self._input.getLogicalTree(), self._filters)

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self._basicElt.jsonSchema()

#    def populate(self, dataList):
#        """This populates the Element Collection. Note that the PROCESSOR is responsible for implementing this Collection's filters"""
#        self._isPopulated = True
#        self._data = []
#        for dataDict in dataList:
#            self._data.append(self._basicElt.populate(dataDict))
#        return self._data


#    def __iter__(self):
#        """Abstract function that returns an iterator over all contents of the Collection that are of the specified type. Providing 'Element' means 'all contents'"""
#        
#        def filteredIterator():
#            for c in self.contents:
#                if all(f.test(c) for f in self.filters):
#                    yield c
#        return filteredIterator()

#    def getAll(elt: Element):
#        "Abstract function that returns only Elements of the specified type."
#        pass

