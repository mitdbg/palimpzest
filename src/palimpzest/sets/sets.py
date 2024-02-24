from palimpzest.elements import *
from palimpzest.operators import *
from palimpzest.datasources import DataDirectory

import json
import hashlib

#####################################################
#
#####################################################
class Set:
    """A Set is set of Elements. It can be iterated over."""
    SET_VERSION = 0.1

    def __init__(self, basicElt, input=None, desc=None, filters=[], aggFunc=None, limit=None):
        self._desc = desc
        self._basicElt = basicElt
        self._input = input
        self._filters = filters
        self._aggFunc = aggFunc
        self._limit = limit

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self._filters])
        return f"{self.__class__.__name__}(basicElt={self._basicElt}, desc={self._desc}, filters={filterStr}, aggFunc={str(self._aggFunc)}, uid={self.universalIdentifier()})"
    
    def serialize(self):
        if self._input is None:
            raise Exception("Cannot create JSON representation of Set because it has no input")

        d = {"version": Set.SET_VERSION, 
             "desc": self._desc, 
             "basicElt": self._basicElt.jsonSchema(), 
             "filters": [f.serialize() for f in self._filters],
             "aggFunc": None if self._aggFunc is None else self._aggFunc.serialize(),
             "limit": self._limit, 
             "input": self._input.serialize()}
        return d

    def deserialize(inputObj):
        if inputObj["version"] != Set.SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        aggFuncStr = inputObj.get("aggFunc", None)
        if aggFuncStr is None:
            aggFunc = None
        else:
            aggFunc = AggregateFunction.deserialize(aggFuncStr)

        limitStr = inputObj.get("limit", None)
        if limitStr is None:
            limit = None
        else:
            limit = int(limitStr)

        return Set(inputObj["basicElt"].jsonSchema(), 
                   input=Set.deserialize(inputObj["input"]), 
                   desc=inputObj["desc"], 
                   aggFunc=aggFunc,
                   limit=limit,
                   filters=[Filter.deserialize(f) for f in inputObj["filters"]])

    def universalIdentifier(self):
        """Return a unique identifier for this Set."""
        d = self.serialize()
        ordered = json.dumps(d, sort_keys=True)
        result = hashlib.sha256(ordered.encode()).hexdigest()
        return result

    def schema(self):
        """The Set's basic element"""
        return self._basicElt

    def filter(self, f: Filter):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        return Set(self._basicElt, input=self, desc="Apply filter(s)", filters=[f])

    def filterByStr(self, filterCondition: str):
        """Add a filter to the Set. This filter will possibly restrict the items that are returned later."""
        f = Filter(filterCondition)
        return self.filter(f)
    
    def aggregate(self, aggFuncDesc: str):
        """Apply an aggregate function to this set"""
        a = AggregateFunction(aggFuncDesc)
        return Set(Number, input=self, desc="Aggregate results", aggFunc=a)
    
    def limit(self, n):
        """Limit the set size to no more than n rows"""
        return Set(self._basicElt, input=self, desc="LIMIT " + str(n), limit=n)
    
    def convert(self, newBasicElt, desc="Convert to new basic element"):
        """Convert the Set to a new basic element."""
        return Set(newBasicElt, input=self, desc=desc)

    def dumpSyntacticTree(self):
        """Return the syntactic tree of this Set."""
        if self._input is None:
            raise Exception("Cannot get syntactic tree of Set because it has no input")
        return (self, self._input.dumpSyntacticTree())

    def getLogicalTree(self):
        """Return the logical tree of operators on Sets."""
        if self._input is None:
            raise Exception("Cannot get logical tree of Set because it has no input")

        # Check to see if there's a cached version of this answer
        uid = self.universalIdentifier()
        if DataDirectory().hasCachedAnswer(uid):
            return CacheScan(self._basicElt, uid)

        # The answer isn't cached, so we have to compute it
        if len(self._filters) > 0:
            return FilteredScan(self._basicElt, self._input.getLogicalTree(), self._filters, targetCacheId=uid)
        elif self._aggFunc is not None:
            return ApplyAggregateFunction(self._basicElt, self._input.getLogicalTree(), self._aggFunc, targetCacheId=uid)
        elif self._limit is not None:
            return LimitScan(self._basicElt, self._input.getLogicalTree(), self._limit, targetCacheId=uid)
        elif not self._basicElt == self._input._basicElt:
            return ConvertScan(self._basicElt, self._input.getLogicalTree(), targetCacheId=uid)
        else:
            return self._input.getLogicalTree()

    def jsonSchema(self):
        """Return the JSON schema for this Set."""
        return self._basicElt.jsonSchema()


def getData(datasetId, basicElt=None):
    """Return a Set of data from the given dataset."""
    return ConcreteDataset(datasetId, targetElt=basicElt)

class ConcreteDataset(Set):
    def __init__(self, uniqName, targetElt=None, desc=None):
        self.uniqName = uniqName

        if not targetElt is None:
            basicElt = targetElt
        else:
            existingDataSet = DataDirectory().getRegisteredDataset(self.uniqName)
            basicElt = Element
            for x in existingDataSet:
                basicElt = x.element
                break

        super().__init__(basicElt, input=None, desc=desc, filters=[])

    def dumpSyntacticTree(self):
        return (self, None)

    def getLogicalTree(self):
        """Return the logical tree of operators on Sets."""
        # REMIND -- this code assumes that all concrete datastores return File objects.
        # If that changes in the future, then this code will have to contact the datastore
        # to figure out the basic element type returned by the datastore.

        uid = self.universalIdentifier()
        if DataDirectory().hasCachedAnswer(uid):
            return CacheScan(self._basicElt, uid)

        existingDataSet = DataDirectory().getRegisteredDataset(self.uniqName)
        existingDataSetElement = None
        for x in existingDataSet:
            existingDataSetElement = x.element
            break

        if existingDataSetElement is None or self._basicElt == existingDataSetElement:
            return BaseScan(self._basicElt, self.uniqName)
        else:
            return ConvertScan(self._basicElt, BaseScan(File, self.uniqName), targetCacheId=uid)

    def serialize(self):
        return {"version": Set.SET_VERSION, 
                "desc": repr(self._desc), 
                "basicElt": repr(self._basicElt),
                "uniqName": self.uniqName}

    def deserialize(inputObj):
        if inputObj["version"] != Set.SET_VERSION:
            raise Exception("Cannot deserialize Set because it is the wrong version")

        return ConcreteDataset(inputObj["basicElt"],
                                 uniqName=inputObj["uniqName"],
                                 desc=eval(inputObj["desc"])) 

