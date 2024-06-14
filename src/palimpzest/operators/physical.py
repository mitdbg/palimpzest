from __future__ import annotations

from palimpzest.constants import *
from palimpzest.corelib import Number, Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.operators import logical

from typing import Any, Callable, Dict, List, Tuple, Optional

import hashlib
import json
import sys
import time

# TYPE DEFINITIONS
DataRecordWithStats = Tuple[DataRecord, RecordOpStats]
DataSourceIteratorFn = Callable[[], DataRecordWithStats]


class ImplementationMeta(type):
    """
    This metaclass is necessary to have the logic: 
    p_op = pz.PhysicalOperator.X
    l_op = pz.logicalOperator.Y
    p_op.implements(l_op) # True or False
    """
    def implements(cls, logical_operator_class):
        return logical_operator_class == cls.implemented_op

class PhysicalOperator(metaclass=ImplementationMeta):
    """
    All implemented physical operators should inherit from this class, and define in the implemented_op variable
    exactly which logical operator they implement. This is necessary for the planner to be able to determine
    which physical operators can be used to implement a given logical operator.
    """

    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    implemented_op = None
    inputSchema = None
    outputSchema = None

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Optional[Schema] = None,
        shouldProfile: bool = False,
        max_workers: int = 1,
    ) -> None:
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema
        self.datadir = DataDirectory()
        self.shouldProfile = shouldProfile
        self.max_workers = max_workers

    def __eq__(self, other: PhysicalOperator) -> bool:
        raise NotImplementedError("Calling __eq__ on abstract method")

    def op_name(self) -> str:
        """Name of the physical operator."""
        return self.__class__.__name__

    def get_op_dict(self):
        raise NotImplementedError("You should implement get_op_dict with op specific parameters")
    
    # NOTE: Simplified the whole op_id flow. Now ops have to implement their get_op_dict method and this method will return the op_id
    def get_op_id(self, plan_position: Optional[int] = None) -> str:
        op_dict = self.get_op_dict()
        if plan_position is not None:
            op_dict["plan_position"] = plan_position

        ordered = json.dumps(op_dict, sort_keys=True)
        hash = hashlib.sha256(ordered.encode()).hexdigest()[:MAX_OP_ID_CHARS]

        op_id = (
            f"{self.op_name()}_{hash}"
            if plan_position is None
            else f"{self.op_name()}_{plan_position}_{hash}"
        )
        return op_id

    def is_hardcoded(self) -> bool:
        """ By default, operators are not hardcoded.
        In those that implement HardcodedConvert or HardcodedFilter, this will return True."""
        return False
        

    def copy(self) -> PhysicalOperator:
        raise NotImplementedError("__copy___ on abstract class")

    def __call__(self, candidate: Any) -> List[DataRecordWithStats]:
        raise NotImplementedError("Using __call__ from abstract method")

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - output_tokens_per_record
        - quality

        The function takes an argument which contains the OperatorCostEstimates
        of the physical operator whose output is the input to this operator.
    
        For the implemented operator. These will be used by the CostEstimator
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOperators need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("CostEstimates from abstract method")

class DataSourcePhysicalOperator(PhysicalOperator):
    """
    By definition, physical operators which implement DataSources don't accept
    a candidate DataRecord as input (because they produce them). Thus, we use
    a slightly modified abstract base class for these operators.
    """
    def naiveCostEstimates(self) -> OperatorCostEstimates:
        """
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - output_tokens_per_record
        - quality
    
        For the implemented operator. These will be used by the CostEstimator
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOperators need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("Abstract method")

    def __call__(self) -> Tuple[List[DataRecord], List[RecordOpStats]]:
        raise NotImplementedError(f"You are calling this method from an abstract class.")

    def __iter__(self) -> DataSourceIteratorFn:
        raise Exception("This method is legacy.")


class MarshalAndScanDataOp(DataSourcePhysicalOperator):

    implemented_op = logical.BaseScan

    def __init__(
        self,
        outputSchema: Schema,
        datasetIdentifier: str,
        num_samples: int = None,
        scan_start_idx: int = 0,
        shouldProfile=False,
    ):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.datasetIdentifier = datasetIdentifier
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.datasetIdentifier == other.datasetIdentifier
            and self.outputSchema == other.outputSchema
            and self.num_samples == other.num_samples
            and self.scan_start_idx == other.scan_start_idx
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + self.datasetIdentifier
            + ")"
        )

    def copy(self):
        return MarshalAndScanDataOp(
            self.outputSchema,
            self.datasetIdentifier,
            self.num_samples,
            self.scan_start_idx,
            self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "datasetIdentifier": self.datasetIdentifier,
        }

    def naiveCostEstimates(self):
        # TODO I have a strong feeling this should belong in the execution layer
        # And datasetIdentified factored out of the logical/physical operators
        cardinality = self.datadir.getCardinality(self.datasetIdentifier) + 1
        size = self.datadir.getSize(self.datasetIdentifier)
        perRecordSizeInKb = (size / float(cardinality)) / 1024.0

        # estimate time spent reading each record
        datasetType = self.datadir.getRegisteredDatasetType(self.datasetIdentifier)
        timePerRecord = (
            LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb
            if datasetType in ["dir", "file"]
            else MEMORY_SCAN_TIME_PER_KB * perRecordSizeInKb
        )

        # for now, assume no cost per record for reading data
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self) -> Tuple[List[DataRecord], List[RecordOpStats]]:
        # Copilot magic
        num_samples = self.num_samples if self.num_samples else float("inf")

        start_time = time.time()
        records = []
        stats = []
        dataset = self.datadir.getRegisteredDataset(self.datasetIdentifier)
        for idx, record in enumerate(dataset):
            end_time = time.time()
            if idx > num_samples:
                break

            if idx < self.scan_start_idx:
                start_time = time.time()
                continue

            kwargs = {
                "op_id": self.get_op_id(),
                "op_name": self.op_name(),
                "op_time": (end_time - start_time),
                "op_cost": 0.0,
                "record_stats": None,
                "op_details": self.__dict__,
            }
            record_op_stats = RecordOpStats.from_record_and_kwargs(record, **kwargs)

            records.append(record)
            stats.append(record_op_stats)

        return records, stats

class CacheScanDataOp(DataSourcePhysicalOperator):
    implemented_op = logical.CacheScan

    def __init__(
        self,
        outputSchema: Schema,
        cacheIdentifier: str,
        num_samples: int = None,
        scan_start_idx: int = 0,
        shouldProfile=False,
    ):
        super().__init__(outputSchema=outputSchema, shouldProfile=shouldProfile)
        self.cacheIdentifier = cacheIdentifier
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.cacheIdentifier == other.cacheIdentifier
            and self.num_samples == other.num_samples
            and self.scan_start_idx == other.scan_start_idx
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + self.cacheIdentifier
            + ")"
        )

    def copy(self):
        return CacheScanDataOp(
            self.outputSchema,
            self.cacheIdentifier,
            self.num_samples,
            self.scan_start_idx,
            self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "datasetIdentifier": self.cacheIdentifier,
        }

    def naiveCostEstimates(self):
        # TODO: at the moment, getCachedResult() looks up a pickled file that stores
        #       the cached data specified by self.cacheIdentifier, opens the file,
        #       and then returns an iterator over records in the pickled file.
        #
        #       I'm guessing that in the future we may want to load the cached data into
        #       the DataDirectory._cache object on __init__ (or in the background) so
        #       that this operation doesn't require a read from disk. If that happens, be
        #       sure to switch LOCAL_SCAN_TIME_PER_KB --> MEMORY_SCAN_TIME_PER_KB; and store
        #       metadata about the cardinality and size of cached data upfront so that we
        #       can access it in constant time.
        #
        #       At a minimum, we could use this function call to load the data into DataManager._cache
        #       since we have to iterate over it anyways; which would cache the data before the __iter__
        #       method below gets called.
        cached_data_info = [
            (1, sys.getsizeof(data))
            for data in self.datadir.getCachedResult(self.cacheIdentifier)
        ]
        cardinality = sum(list(map(lambda tup: tup[0], cached_data_info))) + 1
        size = sum(list(map(lambda tup: tup[1], cached_data_info)))
        perRecordSizeInKb = (size / float(cardinality)) / 1024.0

        # estimate time spent reading each record
        timePerRecord = LOCAL_SCAN_TIME_PER_KB * perRecordSizeInKb

        # for now, assume no cost per record for reading from cache
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=timePerRecord,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self) -> DataSourceIteratorFn:
        # TODO
        def iteratorFn():
            counter = 0
            start_time = time.time()
            for idx, nextCandidate in enumerate(
                self.datadir.getCachedResult(self.cacheIdentifier)
            ):
                end_time = time.time()
                if idx < self.scan_start_idx:
                    start_time = time.time()
                    continue

                kwargs = {
                    "op_id": self.get_op_id(),
                    "op_name": self.op_name(),
                    "op_time": (end_time - start_time),
                    "op_cost": 0.0,
                    "record_stats": None,
                    "op_details": self.__dict__,
                }
                record_op_stats = RecordOpStats.from_record_and_kwargs(nextCandidate, **kwargs)

                yield nextCandidate, record_op_stats

                if self.num_samples:
                    counter += 1
                    if counter >= self.num_samples:
                        break

        return iteratorFn()


class ApplyGroupByOp(PhysicalOperator):
    implemented_op = logical.GroupByAggregate

    def __init__(
        self,
        inputSchema: Schema,
        gbySig: GroupBySig,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=gbySig.outputSchema(),
            shouldProfile=shouldProfile,
        )
        self.inputSchema = inputSchema
        self.gbySig = gbySig
        self.targetCacheId = targetCacheId
        self.shouldProfile = shouldProfile

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.gbySig == other.gbySig
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return f"{self.op_name()}({str(self.gbySig)})"

    def copy(self):
        return ApplyGroupByOp(
            self.inputSchema, self.gbySig, self.targetCacheId, self.shouldProfile
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "gbySig": str(GroupBySig.serialize(self.gbySig)),
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the groupby takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=NAIVE_EST_NUM_GROUPS,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    @staticmethod
    def agg_init(func):
        if func.lower() == "count":
            return 0
        elif func.lower() == "average":
            return (0, 0)
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_merge(func, state, val):
        if func.lower() == "count":
            return state + 1
        elif func.lower() == "average":
            sum, cnt = state
            return (sum + val, cnt + 1)
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_final(func, state):
        if func.lower() == "count":
            return state
        elif func.lower() == "average":
            sum, cnt = state
            return float(sum) / cnt
        else:
            raise Exception("Unknown agg function " + func)

    # TODO: turn this into a __call__ and rely on storing state in class attrs not closure; also return RecordOpStats
    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
        aggState = {}

        @self.profile(
            name="groupby", op_id=self.op_id(), shouldProfile=self.shouldProfile
        )
        def iteratorFn():
            for r in self.source:
                # build group array
                group = ()
                for f in self.gbySig.gbyFields:
                    if not hasattr(r, f):
                        raise TypeError(
                            f"ApplyGroupOp record missing expected field {f}"
                        )
                    group = group + (getattr(r, f),)
                if group in aggState:
                    state = aggState[group]
                else:
                    state = []
                    for fun in self.gbySig.aggFuncs:
                        state.append(ApplyGroupByOp.agg_init(fun))
                for i in range(0, len(self.gbySig.aggFuncs)):
                    fun = self.gbySig.aggFuncs[i]
                    if not hasattr(r, self.gbySig.aggFields[i]):
                        raise TypeError(
                            f"ApplyGroupOp record missing expected field {self.gbySig.aggFields[i]}"
                        )
                    field = getattr(r, self.gbySig.aggFields[i])
                    state[i] = ApplyGroupByOp.agg_merge(fun, state[i], field)
                aggState[group] = state

            gbyFields = self.gbySig.gbyFields
            aggFields = self.gbySig.getAggFieldNames()
            for g in aggState.keys():
                dr = DataRecord(self.gbySig.outputSchema())
                for i in range(0, len(g)):
                    k = g[i]
                    setattr(dr, gbyFields[i], k)
                vals = aggState[g]
                for i in range(0, len(vals)):
                    v = ApplyGroupByOp.agg_final(self.gbySig.aggFuncs[i], vals[i])
                    setattr(dr, aggFields[i], v)
                if shouldCache:
                    datadir.appendCache(self.targetCacheId, dr)
                yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyCountAggregateOp(PhysicalOperator):
    implemented_op = logical.ApplyAggregateFunction

    def __init__(
        self,
        inputSchema: Schema,
        aggFunction: AggregateFunction,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema, outputSchema=Number, shouldProfile=shouldProfile
        )
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunction == other.aggFunction
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Function: "
            + str(self.aggFunction)
            + ")"
        )

    def copy(self):
        return ApplyCountAggregateOp(
            inputSchema=self.inputSchema,
            aggFunction=self.aggFunction,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "aggFunction": str(self.aggFunction)
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    # TODO: turn this into a __call__ and rely on storing state in class attrs not closure; also return RecordOpStats
    def __iter__(self):
        raise NotImplementedError("TODO method")
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="count", shouldProfile=self.shouldProfile)
        def iteratorFn():
            counter = 0
            for record in self.source:
                counter += 1

            # NOTE: this will set the parent_uuid to be the uuid of the final source record;
            #       this is ideal for computing the op_time of the count operation, but maybe
            #       we should set this DataRecord as having multiple parents in the future
            dr = DataRecord(Number, parent_uuid=record._uuid)
            dr.value = counter
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


# TODO: coalesce into base class w/ApplyCountAggregateOp and simply override __call__ methods in base classes
# GV: What if we keep the two separate and have two different logical operators? 
class ApplyAverageAggregateOp(PhysicalOperator):
    implemented_op = logical.ApplyAggregateFunction

    def __init__(
        self,
        inputSchema: Schema,
        aggFunction: AggregateFunction,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema, outputSchema=Number, shouldProfile=shouldProfile
        )
        self.aggFunction = aggFunction
        self.targetCacheId = targetCacheId

        if not inputSchema == Number:
            raise Exception("Aggregate function AVERAGE is only defined over Numbers")

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.aggFunction == other.aggFunction
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Function: "
            + str(self.aggFunction)
            + ")"
        )

    def copy(self):
        return ApplyAverageAggregateOp(
            inputSchema=self.inputSchema,
            aggFunction=self.aggFunction,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "aggFunction": str(self.aggFunction)
        }

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    # TODO: turn this into a __call__ and rely on storing state in class attrs not closure; also return RecordOpStats
    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="average", shouldProfile=self.shouldProfile)
        def iteratorFn():
            sum = 0
            counter = 0
            for nextCandidate in self.source:
                try:
                    sum += int(nextCandidate.value)
                    counter += 1
                except:
                    pass

            # NOTE: this will set the parent_uuid to be the uuid of the final source record;
            #       this is ideal for computing the op_time of the count operation, but maybe
            #       we should set this DataRecord as having multiple parents in the future
            dr = DataRecord(Number, parent_uuid=nextCandidate._uuid)
            dr.value = sum / float(counter)
            if shouldCache:
                datadir.appendCache(self.targetCacheId, dr)
            yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class LimitScanOp(PhysicalOperator):
    implemented_op = logical.LimitScan

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Schema,
        limit: int,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=outputSchema,
            shouldProfile=shouldProfile,
        )
        self.limit = limit
        self.targetCacheId = targetCacheId

    def __eq__(self, other: PhysicalOperator):
        return (
            isinstance(other, self.__class__)
            and self.limit == other.limit
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            f"{self.op_name()}("
            + str(self.outputSchema)
            + ", "
            + "Limit: "
            + str(self.limit)
            + ")"
        )

    def copy(self):
        return LimitScanOp(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            limit=self.limit,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def get_op_dict(self):
        return {
            "operator": self.op_name(),
            "outputSchema": str(self.outputSchema),
            "limit": self.limit,
        }


    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the limit takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=min(self.limit, source_op_cost_estimates.cardinality),
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    # TODO: turn this into a __call__ and rely on storing state in class attrs not closure; also return RecordOpStats
    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="limit", shouldProfile=self.shouldProfile)
        def iteratorFn():
            counter = 0
            for nextCandidate in self.source:
                if shouldCache:
                    datadir.appendCache(self.targetCacheId, nextCandidate)
                yield nextCandidate

                counter += 1
                if counter >= self.limit:
                    break

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()
