from __future__ import annotations

from palimpzest.corelib import Schema
from palimpzest.elements import *

from typing import List

from palimpzest.datamanager import DataDirectory


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets. Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    - LimitScan (scans up to N records from a Set)
    - ApplyAggregateFunction (applies an aggregation on the Set)

    Every logical operator must declare the getParameters() method, which returns a dictionary of parameters that are used to implement its physical operator.
    """

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
    ):
        self.inputSchema = inputSchema
        self.outputSchema = outputSchema

    def getParameters(self) -> dict:
        raise NotImplementedError("Abstract method")
    
    def __str__(self) -> str:
        raise NotImplementedError("Abstract method")

    def copy(self) -> LogicalOperator:
        raise NotImplementedError("Abstract method")

    def logical_op_id(self) -> str:
        raise NotImplementedError("Abstract method")


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""

    def __init__(
        self,
        cardinality: str = "oneToOne",
        image_conversion: bool = False,
        depends_on: List[str] = None,
        desc: str = None,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cardinality = cardinality
        self.image_conversion = image_conversion
        self.depends_on = depends_on
        self.desc = desc
        self.targetCacheId = targetCacheId

        # TODO: we will run into trouble here in a scenario like the following:
        # - Schema A inherits from TextFile
        # - Schema B inherits from pz.Schema
        # - Schema C inherits from TextFile
        # - convert A -> B happens first
        # - convert B -> C happens second <-- issue is here
        #
        # the diff. in fieldNames between C and B will include things like "contents"
        # which come from pz.TextFile, and it will cause the second convert to try to
        # (re)compute the "contents" field.
        #
        # compute generated fields as set of fields in outputSchema that are not in inputSchema
        self.generated_fields = [
            field
            for field in self.outputSchema.fieldNames()
            if field not in self.inputSchema.fieldNames()
        ]

    def __str__(self):
        return f"ConvertScan({self.inputSchema} -> {str(self.outputSchema)},{str(self.desc)})"

    def copy(self):
        return ConvertScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            depends_on=self.depends_on,
            desc=self.desc,
            targetCacheId=self.targetCacheId,
        )

    def logical_op_id(self):
        return f"{self.__class__.__name__}"
    
    def getParameters(self) -> dict:
        return {
            "inputSchema": self.inputSchema,
            "outputSchema": self.outputSchema,
            "cardinality": self.cardinality,
            "image_conversion": self.image_conversion,
            "depends_on": self.depends_on,
            "desc": self.desc,
            "targetCacheId": self.targetCacheId,
            "generated_fields": self.generated_fields
            }

class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""

    def __init__(self, cachedDataIdentifier: str, *args, **kwargs):
        kwargs["inputSchema"] = None

        super().__init__(None, *args, **kwargs)
        self.cachedDataIdentifier = cachedDataIdentifier

    def __str__(self):
        return f"CacheScan({str(self.outputSchema)},{str(self.cachedDataIdentifier)})"

    def copy(self):
        return CacheScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            cachedDataIdentifier=self.cachedDataIdentifier,
        )

    def getParameters(self) -> dict:
        return {
            "outputSchema": self.outputSchema,
            "cachedDataIdentifier": self.cachedDataIdentifier
            }

# NOTE: I feel we should remove datasetIdentifier from both the logical and physical operator. My argument is that the logical BaseScan is the same no matter what the datasetidentifier is. The datasetIdentifier is an execution-level  detail. Think about having two exactly equal workloads: except one is defined on folder enron-eval-tiny, one is defined on enron-eval-full. Why should the logical and physical *plans* be different ? The *execution* will be different, much like running a Filter on two different data items will differ.
class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""

    def __init__(self, datasetIdentifier: str, *args, **kwargs):
        kwargs["inputSchema"] = None
        super().__init__(*args, **kwargs)
        self.datasetIdentifier = datasetIdentifier
        self.dataset_type = DataDirectory().getRegisteredDatasetType(datasetIdentifier)

    def __str__(self):
        return f"BaseScan({self.datasetIdentifier},{str(self.outputSchema)})"

    def copy(self):
        return BaseScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            datasetIdentifier=self.datasetIdentifier,
        )
    
    def getParameters(self) -> dict:
        return {
            "outputSchema": self.outputSchema,
            "dataset_type": self.dataset_type
            }


class LimitScan(LogicalOperator):
    def __init__(self, limit: int, targetCacheId: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"LimitScan({str(self.inputSchema)}, {str(self.outputSchema)})"

    def copy(self):
        return LimitScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            limit=self.limit,
            targetCacheId=self.targetCacheId,
        )

    def getParameters(self) -> dict:
        return {
            "inputSchema": self.inputSchema,
            "outputSchema": self.outputSchema,
            "limit": self.limit,
            "targetCacheId": self.targetCacheId
            }


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""

    def __init__(
        self,
        filter: Filter,
        depends_on: List[str] = None,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.filter = filter
        self.depends_on = depends_on
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"FilteredScan({str(self.outputSchema)}, {str(self.filter)})"

    def copy(self):
        return FilteredScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            filter=self.filter,
            depends_on=self.depends_on,
            targetCacheId=self.targetCacheId,
        )
    
    def getParameters(self) -> dict:
        return {
            "inputSchema":self.inputSchema,
            "outputSchema":self.outputSchema,
            "filter":self.filter,
            }


class GroupByAggregate(LogicalOperator):
    def __init__(
        self,
        gbySig: GroupBySig,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        (valid, error) = gbySig.validateSchema(self.inputSchema)
        if not valid:
            raise TypeError(error)
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"GroupBy({GroupBySig.serialize(self.gbySig)})"

    def copy(self):
        return GroupByAggregate(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            gbySig=self.gbySig,
            targetCacheId=self.targetCacheId,
        )

    def getParameters(self) -> dict:
        return {
            "inputSchema": self.inputSchema,
            "gbySig": self.gbySig,
            "targetCacheId": self.targetCacheId
            }


class ApplyAggregateFunction(LogicalOperator):
    """ApplyAggregateFunction is a logical operator that applies a function to the input set and yields a single result.
    This is a base class that has to be further specialized to implement specific aggregation functions.
    """
    aggregationFunction: None

    def __init__(
        self,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"{self.__class__.__name__}(function: {str(self.aggregationFunction)})"

    def copy(self):
        return ApplyAggregateFunction(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            aggregationFunction=self.aggregationFunction,
            targetCacheId=self.targetCacheId,
        )
    
    def getParameters(self) -> dict:
        return {
            "inputSchema":self.inputSchema,
            "aggFunction":self.aggregationFunction,
            "targetCacheId":self.targetCacheId,
        }

class ApplyCountAggregateFunction(ApplyAggregateFunction):
    aggregationFunction = AggregateFunction.COUNT

class ApplyAverageAggregateFunction(ApplyAggregateFunction):
    aggregationFunction = AggregateFunction.AVERAGE