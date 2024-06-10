from __future__ import annotations

from palimpzest.corelib import Schema
from palimpzest.elements import *

from typing import List


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets. Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    - LimitScan (scans up to N records from a Set)
    - ApplyAggregateFunction (applies an aggregation on the Set)
    """

    def __init__(
        self,
        inputSchema: Schema,
        outputSchema: Schema,
    ):
        self.inputSchema = inputSchema
        self.outputSchema = outputSchema

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


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""

    def __init__(self, datasetIdentifier: str, *args, **kwargs):
        kwargs["inputSchema"] = None

        super().__init__(*args, **kwargs)
        self.datasetIdentifier = datasetIdentifier

    def __str__(self):
        return f"BaseScan({str(self.outputSchema)},{self.datasetIdentifier})"

    def copy(self):
        return BaseScan(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            datasetIdentifier=self.datasetIdentifier,
        )


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


class ApplyAggregateFunction(LogicalOperator):
    """ApplyAggregateFunction is a logical operator that applies a function to the input set and yields a single result."""

    def __init__(
        self,
        aggregationFunction: AggregateFunction,
        targetCacheId: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregationFunction = aggregationFunction
        self.targetCacheId = targetCacheId

    def __str__(self):
        return f"ApplyAggregateFunction(function: {str(self.aggregationFunction)})"

    def copy(self):
        return ApplyAggregateFunction(
            inputSchema=self.inputSchema,
            outputSchema=self.outputSchema,
            aggregationFunction=self.aggregationFunction,
            targetCacheId=self.targetCacheId,
        )
