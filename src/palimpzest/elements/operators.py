from .elements import *
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

class BaseScan(LogicalOperator):
    """A ConcreteScan is a logical operator that represents a scan of a particular data source."""
    def __init__(self, outputElementType):
        super().__init__(outputElementType, None)

    def __str__(self):
        return "BaseScan(" + str(self.outputElementType) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def getPhysicalTree(self):
        return InduceFromCandidateOp(self.outputElementType)

class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""
    def __init__(self, outputElementType, inputOp, filters):
        super().__init__(outputElementType, inputOp.outputElementType)
        self.inputOp = inputOp
        self.filters = filters

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "FilteredScan(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def getPhysicalTree(self):
        return FilterCandidateOp(self.outputElementType, self.inputOp.getPhysicalTree(), self.filters)
