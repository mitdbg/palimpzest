from palimpzest.elements import *
from .physical import *

class LogicalOperator:
    """A logical operator is an operator that operates on sets. Right now it can be a FilteredScan or a ConcreteScan."""
    def __init__(self, outputElementType, inputElementType):
        self.outputElementType = outputElementType
        self.inputElementType = inputElementType

    def dumpLogicalTree(self):
        raise NotImplementedError("Abstract method")
    
    def _getPhysicalTree(self, strategy=None):
        raise NotImplementedError("Abstract method")

    def createPhysicalPlan(self):
        """Create the physical tree of operators."""
        plan1 = self._getPhysicalTree(strategy=PhysicalOp.LOCAL_PLAN)
        plan2 = self._getPhysicalTree(strategy=PhysicalOp.REMOTE_PLAN)

        plan1Cost = plan1.estimateCost()
        plan2Cost = plan2.estimateCost()

        totalTime1 = plan1Cost["timePerElement"] * plan1Cost["cardinality"] + plan1Cost["startupTime"]
        totalTime2 = plan2Cost["timePerElement"] * plan2Cost["cardinality"] + plan2Cost["startupTime"]
        totalPrice1 = plan1Cost["costPerElement"] * plan1Cost["cardinality"] + plan1Cost["startupCost"]
        totalPrice2 = plan2Cost["costPerElement"] * plan2Cost["cardinality"] + plan2Cost["startupCost"]

        if totalTime1 < totalTime2:
            return totalTime1, totalPrice1, plan1Cost["cardinality"], plan1 
        else:
            return totalTime2, totalPrice2, plan2Cost["cardinality"], plan2


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""
    def __init__(self, outputElementType, inputOp, targetCacheId=None):
        super().__init__(outputElementType, inputOp.outputElementType)
        self.inputOp = inputOp
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "ConvertScan(" + str(self.inputElementType) +", " + str(self.outputElementType) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy=None):
        # If the input is in core, and the output is NOT in core but its superclass is, then we should do a
        # 2-stage conversion. This will maximize chances that there is a pre-existing conversion to the superclass
        # in the known set of functions
        intermediateOutputElement = self.outputElementType
        while not intermediateOutputElement == Element and not PhysicalOp.solver.easyConversionAvailable(intermediateOutputElement, self.inputElementType):
            intermediateOutputElement = intermediateOutputElement.__bases__[0]

        if intermediateOutputElement == Element or intermediateOutputElement == self.outputElementType:
            if DataDirectory().config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputElementType, self.inputOp._getPhysicalTree(strategy=strategy), targetCacheId=self.targetCacheId)
            else:
                return InduceFromCandidateOp(self.outputElementType, self.inputOp._getPhysicalTree(strategy=strategy), targetCacheId=self.targetCacheId)
        else:
            if DataDirectory().config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputElementType, ParallelInduceFromCandidateOp(intermediateOutputElement, self.inputOp._getPhysicalTree(strategy=strategy)), targetCacheId=self.targetCacheId)
            else:
                return InduceFromCandidateOp(self.outputElementType, 
                                             InduceFromCandidateOp(
                                                 intermediateOutputElement, 
                                                 self.inputOp._getPhysicalTree(strategy=strategy)),
                                             targetCacheId=self.targetCacheId)

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

    def _getPhysicalTree(self, strategy=None):
        return CacheScanDataOp(self.outputElementType, self.cachedDataIdentifier)

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

    def _getPhysicalTree(self, strategy=None):
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

    def _getPhysicalTree(self, strategy=None):
        return FilterCandidateOp(self.outputElementType, self.inputOp._getPhysicalTree(strategy=strategy), self.filters, targetCacheId=self.targetCacheId)
