from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.operators import (
    CacheScanDataOp,
    FilterCandidateOp,
    InduceFromCandidateOp,
    MarshalAndScanDataOp,
    ParallelFilterCandidateOp,
    ParallelInduceFromCandidateOp,
    PhysicalOp,
)

from __future__ import annotations
from typing import Tuple


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets. Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    """
    def __init__(self, outputSchema: Schema, inputSchema: Schema):
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema

    def dumpLogicalTree(self) -> Tuple[LogicalOperator, LogicalOperator]:
        raise NotImplementedError("Abstract method")

    def _getPhysicalTree(self, strategy: str=None) -> PhysicalOp:
        raise NotImplementedError("Abstract method")

    def createPhysicalPlans(self) -> Tuple[float, float, float, PhysicalOp]:
        """Return a set of physical trees of operators."""
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
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "ConvertScan(" + str(self.inputSchema) +", " + str(self.outputSchema) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy: str=None):
        # If the input is in core, and the output is NOT in core but its superclass is, then we should do a
        # 2-stage conversion. This will maximize chances that there is a pre-existing conversion to the superclass
        # in the known set of functions
        intermediateSchema = self.outputSchema
        while not intermediateSchema == Schema and not PhysicalOp.solver.easyConversionAvailable(intermediateSchema, self.inputSchema):
            intermediateSchema = intermediateSchema.__bases__[0]

        if intermediateSchema == Schema or intermediateSchema == self.outputSchema:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy), targetCacheId=self.targetCacheId)
            else:
                return InduceFromCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy), targetCacheId=self.targetCacheId)
        else:
            if DataDirectory().current_config.get("parallel") == True:
                return ParallelInduceFromCandidateOp(self.outputSchema, ParallelInduceFromCandidateOp(intermediateSchema, self.inputOp._getPhysicalTree(strategy=strategy)), targetCacheId=self.targetCacheId)
            else:
                return InduceFromCandidateOp(self.outputSchema, 
                                             InduceFromCandidateOp(
                                                 intermediateSchema, 
                                                 self.inputOp._getPhysicalTree(strategy=strategy)),
                                             targetCacheId=self.targetCacheId)

class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""
    def __init__(self, outputSchema: Schema, cachedDataIdentifier: str):
        super().__init__(outputSchema, None)
        self.cachedDataIdentifier = cachedDataIdentifier

    def __str__(self):
        return "CacheScan(" + str(self.outputSchema) + ", " + str(self.cachedDataIdentifier) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None):
        return CacheScanDataOp(self.outputSchema, self.cachedDataIdentifier)

class BaseScan(LogicalOperator):
    """A ConcreteScan is a logical operator that represents a scan of a particular data source."""
    def __init__(self, outputSchema: Schema, concreteDatasetIdentifier: str):
        super().__init__(outputSchema, None)
        self.concreteDatasetIdentifier = concreteDatasetIdentifier

    def __str__(self):
        return "BaseScan(" + str(self.outputSchema) + ", " + self.concreteDatasetIdentifier + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, None)

    def _getPhysicalTree(self, strategy: str=None):
        return MarshalAndScanDataOp(self.outputSchema, self.concreteDatasetIdentifier)

class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""
    def __init__(self, outputSchema: Schema, inputOp: LogicalOperator, filter: Filter, targetCacheId: str=None):
        super().__init__(outputSchema, inputOp.outputSchema)
        self.inputOp = inputOp
        self.filter = filter
        self.targetCacheId = targetCacheId

    def __str__(self):
        return "FilteredScan(" + str(self.outputSchema) + ", " + "Filters: " + str(self.filter) + ")"

    def dumpLogicalTree(self):
        """Return the logical tree of this LogicalOperator."""
        return (self, self.inputOp.dumpLogicalTree())

    def _getPhysicalTree(self, strategy=None):
        if DataDirectory().current_config.get("parallel") == True:
            return ParallelFilterCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy), self.filter, targetCacheId=self.targetCacheId)
        else:
            return FilterCandidateOp(self.outputSchema, self.inputOp._getPhysicalTree(strategy=strategy), self.filter, targetCacheId=self.targetCacheId)
