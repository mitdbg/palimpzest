from palimpzest.elements import *
from palimpzest.operators import *

#####################################################
#
#####################################################
class Set:
    """A Set is set of Elements. It can be iterated over."""
    SET_VERSION = 0.1

    def __init__(self, basicElt, input=None, desc=None, filters=[]):
        self._desc = desc
        self._basicElt = basicElt
        self._input = input
        self._filters = filters

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self._filters])
        return f"{self.__class__.__name__}(basicElt={self._basicElt}, desc={self._desc}, filters={filterStr})"

    def serialize(self):
        if self._input is None:
            raise Exception("Cannot create JSON representation of Set because it has no input")

        return {"version": SET_VERSION, 
             "desc": repr(self._desc), 
             "basicElt": repr(self._basicElt), 
             "filters": repr(self._filters), 
             "input": self._input.serialize()}

    def deserialize(inputObj):
        if inputObj["version"] != SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        return Set(inputObj["basicElt"], 
                   input=Set.deserialize(inputObj["input"]), 
                   desc=eval(inputObj["desc"]), 
                   filters=eval(inputObj["filters"]))

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
        if self._input is None:
            raise Exception("Cannot get syntactic tree of Set because it has no input")
        return (self, self._input.dumpSyntacticTree())

    def getLogicalTree(self):
        """Return the logical tree of operators on Sets."""
        if self._input is None:
            raise Exception("Cannot get logical tree of Set because it has no input")

        if len(self._filters) >= 0 and not self._basicElt == self._input._basicElt:
            return FilteredScan(self._basicElt, ConvertScan(self._basicElt, self._input.getLogicalTree()), self._filters)
        elif len(self._filters) == 0 and not self._basicElt == self._input._basicElt:
            return ConvertScan(self._basicElt, self._input.getLogicalTree())
        elif len(self._filters) >= 0 and self._basicElt == self._input._basicElt:
            return FilteredScan(self._basicElt, self._input.getLogicalTree(), self._filters)
        else:
            return self._input.getLogicalTree()

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self._basicElt.jsonSchema()

class ConcreteDataset(Set):
    def __init__(self, basicElt, uniqName, desc=None):
        super().__init__(basicElt, input=None, desc=desc, filters=[])
        self.uniqName = uniqName

    def dumpSyntacticTree(self):
        return (self, None)

    def getLogicalTree(self):
        """Return the logical tree of operators on Sets."""
        return BaseScan(self._basicElt, self.uniqName)

    def serialize(self):
        return {"version": SET_VERSION, 
                "desc": repr(self._desc), 
                "basicElt": repr(self._basicElt),
                "uniqName": self.uniqName}

    def deserialize(inputObj):
        if inputObj["version"] != SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        return ConcreteDataset(inputObj["basicElt"],
                                 uniqName=inputObj["uniqName"],
                                 desc=eval(inputObj["desc"])) 

