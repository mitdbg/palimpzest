from palimpzest.elements import *
from .physical import *

class LogicalOperator:
    """A logical operator is an operator that operates on sets. Right now it can be a FilteredScan or a ConcreteScan."""
    def __init__(self, outputElementType, inputElementType):
        self.outputElementType = outputElementType
        self.inputElementType = inputElementType

    def dumpLogicalTree(self):
        raise NotImplementedError("Abstract method")
    
    def getPhysicalTree(self):
        raise NotImplementedError("Abstract method")

class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""
    def __init__(self, outputElementType, inputOp):
        super().__init__(outputElementType, inputOp.outputElementType)
        self.inputOp = inputOp

    def __str__(self):
        return "ConvertScan(" + str(self.inputElementType) +", " + str(self.outputElementType) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def getPhysicalTree(self):
        return InduceFromCandidateOp(self.outputElementType, self.inputOp.getPhysicalTree())

class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached answer."""
    def __init__(self, outputElementType, cachedDataIdentifier):
        super().__init__(outputElementType, None)
        self.cachedDataIdentifier = cachedDataIdentifier

    def __str__(self):
        return "CacheScan(" + str(self.outputElementType) + ", " + str(self.cachedDataIdentifier) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def getPhysicalTree(self):
        return MarshalAndScanDataOp(self.outputElementType, self.cachedDataIdentifier)

class BaseScan(LogicalOperator):
    """A ConcreteScan is a logical operator that represents a scan of a particular data source."""
    def __init__(self, outputElementType, concreteDatasetIdentifier):
        super().__init__(outputElementType, None)
        self.concreteDatasetIdentifier = concreteDatasetIdentifier

    def __str__(self):
        return "BaseScan(" + str(self.outputElementType) + ", " + self.concreteDatasetIdentifier + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def getPhysicalTree(self):
        return MarshalAndScanDataOp(self.outputElementType, self.concreteDatasetIdentifier)

class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""
    def __init__(self, outputElementType, inputOp, filters, targetCacheId=None):
        super().__init__(outputElementType, inputOp.outputElementType)
        self.inputOp = inputOp
        self.filters = filters
        self.targetCacheId = targetCacheId

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "FilteredScan(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def getPhysicalTree(self):
        return FilterCandidateOp(self.outputElementType, self.inputOp.getPhysicalTree(), self.filters, targetCacheId=self.targetCacheId)
