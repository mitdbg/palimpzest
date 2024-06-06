from __future__ import annotations

import math

from palimpzest.constants import *
from palimpzest.corelib import ImageFile, Number, Schema
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.profiler import OperatorStats, Profiler, StatsProcessor

from typing import Any, Callable, Dict, List, Tuple, Union, Optional

import pandas as pd

import concurrent
import hashlib
import json
import sys

# DEFINITIONS
MAX_ID_CHARS = 10
IteratorFn = Callable[[], DataRecord]


class PhysicalOp:
    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    # synthesizedFns = {}
    # solver = Solver(verbose=LOG_LLM_OUTPUT)

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Optional[Schema] = None,
        shouldProfile=False,
        max_workers: int = 1,
    ) -> None:
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema
        self.datadir = DataDirectory()
        self.shouldProfile = shouldProfile
        self.plan_idx = None
        self.max_workers = max_workers

        # NOTE: this must be overridden in each physical operator's __init__ method;
        #       we have to do it their b/c the opId() (which is an argument to the
        #       profiler's constructor) may not be valid until the physical operator
        #       has initialized all of its member fields
        self.profiler = None

    def __eq__(self, other: PhysicalOp) -> bool:
        raise NotImplementedError("Abstract method")

    def opId(self) -> str:
        raise NotImplementedError("Abstract method")

    def is_hardcoded(self) -> bool:
        if self.inputSchema is None:
            return True
        return (self.outputSchema, self.inputSchema) in self.solver._hardcodedFns

    def copy(self) -> PhysicalOp:
        raise NotImplementedError

    def dumpPhysicalTree(self) -> Tuple[PhysicalOp, Union[PhysicalOp, None]]:
        raise NotImplementedError("Legacy method")
        """Return the physical tree of operators."""
        if self.inputSchema is None:
            return (self, None)
        return (self, self.source.dumpPhysicalTree())

    def setPlanIdx(self, idx) -> None:
        raise NotImplementedError("Legacy method")
        self.plan_idx = idx
        if self.source is not None:
            self.source.setPlanIdx(idx)

    def getProfilingData(self) -> OperatorStats:
        # simply return stats for this operator if there is no source
        if self.shouldProfile and self.source is None:
            return self.profiler.get_data()

        # otherwise, fetch the source operator's stats first, and then return
        # the current operator's stats w/a copy of its sources' stats
        elif self.shouldProfile:
            source_operator_stats = self.source.getProfilingData()
            operator_stats = self.profiler.get_data()
            operator_stats.source_op_stats = source_operator_stats
            return operator_stats

        # raise an exception if this method is called w/out profiling turned on
        else:
            raise Exception(
                "Profiling was not turned on; please ensure shouldProfile=True when executing plan."
            )

    def estimateCost(
        self, cost_estimate_sample_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Returns dict of time, cost, and quality metrics."""
        raise NotImplementedError("Abstract method")


class MarshalAndScanDataOp(PhysicalOp):
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, MarshalAndScanDataOp)
            and self.datasetIdentifier == other.datasetIdentifier
            and self.num_samples == other.num_samples
            and self.scan_start_idx == other.scan_start_idx
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return (
            "MarshalAndScanDataOp("
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

    def opId(self):
        d = {
            "operator": "MarshalAndScanDataOp",
            "outputSchema": str(self.outputSchema),
            "datasetIdentifier": self.datasetIdentifier,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        cardinality = self.datadir.getCardinality(self.datasetIdentifier) + 1
        size = self.datadir.getSize(self.datasetIdentifier)
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        # if we have sample data, use it to get a better estimate of the timePerElement
        # and the output tokens per element
        timePerElement, op_filter = None, "op_name == 'base_scan'"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            timePerElement = cost_est_data[op_filter]["time_per_record"]
        else:
            # estimate time spent reading each record
            datasetType = self.datadir.getRegisteredDatasetType(self.datasetIdentifier)
            timePerElement = (
                LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
                if datasetType in ["dir", "file"]
                else MEMORY_SCAN_TIME_PER_KB * perElementSizeInKb
            )

        # NOTE: downstream operators will ignore this estimate if they have a cost_estimate dict.
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality))  # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT  # fraction of the element which is provided in context
            * BYTES_TO_TOKENS  # convert bytes to tokens
        )

        # assume no cost for reading data
        usdPerElement = 0

        costEst = {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

        return costEst, {"cumulative": costEst, "thisPlan": costEst, "subPlan": None}

    def __iter__(self) -> IteratorFn:
        @self.profile(name="base_scan", shouldProfile=self.shouldProfile)
        def iteratorFn():
            counter = 0
            for idx, nextCandidate in enumerate(
                self.datadir.getRegisteredDataset(self.datasetIdentifier)
            ):
                if idx < self.scan_start_idx:
                    continue

                yield nextCandidate

                if self.num_samples:
                    counter += 1
                    if counter >= self.num_samples:
                        break

        return iteratorFn()


class CacheScanDataOp(PhysicalOp):
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, CacheScanDataOp)
            and self.cacheIdentifier == other.cacheIdentifier
            and self.num_samples == other.num_samples
            and self.scan_start_idx == other.scan_start_idx
            and self.outputSchema == other.outputSchema
        )

    def __str__(self):
        return (
            "CacheScanDataOp("
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

    def opId(self):
        d = {
            "operator": "CacheScanDataOp",
            "outputSchema": str(self.outputSchema),
            "cacheIdentifier": self.cacheIdentifier,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
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
        perElementSizeInKb = (size / float(cardinality)) / 1024.0

        # if we have sample data, use it to get a better estimate of the timePerElement
        # and the output tokens per element
        timePerElement, op_filter = None, "op_name == 'cache_scan'"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            timePerElement = cost_est_data[op_filter]["time_per_record"]
        else:
            # estimate time spent reading each record
            timePerElement = LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb

        # assume no cost for reading data
        usdPerElement = 0

        # NOTE: downstream operators will ignore this estimate if they have a cost_estimate dict.
        # estimate per-element number of tokens output by this operator
        estOutputTokensPerElement = (
            (size / float(cardinality))  # per-element size in bytes
            * ELEMENT_FRAC_IN_CONTEXT  # fraction of the element which is provided in context
            * BYTES_TO_TOKENS  # convert bytes to tokens
        )

        costEst = {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "usdPerElement": usdPerElement,
            "cumulativeTimePerElement": timePerElement,
            "cumulativeUSDPerElement": usdPerElement,
            "totalTime": timePerElement * cardinality,
            "totalUSD": usdPerElement * cardinality,
            "estOutputTokensPerElement": estOutputTokensPerElement,
            "quality": 1.0,
        }

        return costEst, {"cumulative": costEst, "thisPlan": costEst, "subPlan": None}

    def __iter__(self) -> IteratorFn:
        @self.profile(name="cache_scan", shouldProfile=self.shouldProfile)
        def iteratorFn():
            # NOTE: see comment in `estimateCost()`
            counter = 0
            for idx, nextCandidate in enumerate(
                self.datadir.getCachedResult(self.cacheIdentifier)
            ):
                if idx < self.scan_start_idx:
                    continue

                yield nextCandidate

                if self.num_samples:
                    counter += 1
                    if counter >= self.num_samples:
                        break

        return iteratorFn()


def agg_init(func):
    if func.lower() == "count":
        return 0
    elif func.lower() == "average":
        return (0, 0)
    else:
        raise Exception("Unknown agg function " + func)


def agg_merge(func, state, val):
    if func.lower() == "count":
        return state + 1
    elif func.lower() == "average":
        sum, cnt = state
        return (sum + val, cnt + 1)
    else:
        raise Exception("Unknown agg function " + func)


def agg_final(func, state):
    if func.lower() == "count":
        return state
    elif func.lower() == "average":
        sum, cnt = state
        return float(sum) / cnt
    else:
        raise Exception("Unknown agg function " + func)


class ApplyGroupByOp(PhysicalOp):
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

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, ApplyGroupByOp)
            and self.gbySig == other.gbySig
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return str(self.gbySig)

    def copy(self):
        return ApplyGroupByOp(
            self.source, self.gbySig, self.targetCacheId, self.shouldProfile
        )

    def opId(self):
        d = {
            "operator": "ApplyGroupByOp",
            "source": self.source.opId(),
            "gbySig": str(GroupBySig.serialize(self.gbySig)),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputEstimates, subPlanCostEst = self.source.estimateCost()

        outputEstimates = {**inputEstimates}
        outputEstimates["cardinality"] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates, {
            "cumulative": outputEstimates,
            "thisPlan": {},
            "subPlan": subPlanCostEst,
        }

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
        aggState = {}

        @self.profile(
            name="groupby", op_id=self.opId(), shouldProfile=self.shouldProfile
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
                        state.append(agg_init(fun))
                for i in range(0, len(self.gbySig.aggFuncs)):
                    fun = self.gbySig.aggFuncs[i]
                    if not hasattr(r, self.gbySig.aggFields[i]):
                        raise TypeError(
                            f"ApplyGroupOp record missing expected field {self.gbySig.aggFields[i]}"
                        )
                    field = getattr(r, self.gbySig.aggFields[i])
                    state[i] = agg_merge(fun, state[i], field)
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
                    v = agg_final(self.gbySig.aggFuncs[i], vals[i])
                    setattr(dr, aggFields[i], v)
                if shouldCache:
                    datadir.appendCache(self.targetCacheId, dr)
                yield dr

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyCountAggregateOp(PhysicalOp):
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, ApplyCountAggregateOp)
            and self.aggFunction == other.aggFunction
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            "ApplyCountAggregateOp("
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

    def opId(self):
        raise NotImplementedError("Legacy method")
        d = {
            "operator": "ApplyCountAggregateOp",
            "source": self.source.opId(),
            "aggFunction": str(self.aggFunction),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        # get input estimates and pass through to output
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record timing info for this operator, which can be used
        # to improve timing related estimates
        op_filter = "(op_name == 'count')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # compute estimates
            time_per_record = cost_est_data[op_filter]["time_per_record"]

            # output cardinality for an aggregate will be 1
            cardinality = 1

            # update cardinality, timePerElement and related stats
            outputEstimates["cardinality"] = cardinality
            outputEstimates["timePerElement"] = time_per_record
            outputEstimates["cumulativeTimePerElement"] = (
                inputEstimates["cumulativeTimePerElement"] + time_per_record
            )
            outputEstimates["totalTime"] = (
                cardinality * time_per_record + inputEstimates["totalTime"]
            )

            return outputEstimates, {
                "cumulative": outputEstimates,
                "thisPlan": {"time_per_record": time_per_record},
                "subPlan": subPlanCostEst,
            }

        # output cardinality for an aggregate will be 1
        outputEstimates["cardinality"] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates, {
            "cumulative": outputEstimates,
            "thisPlan": {},
            "subPlan": subPlanCostEst,
        }

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


# TODO: remove in favor of users in-lining lambdas
class ApplyUserFunctionOp(PhysicalOp):
    def __init__(
        self,
        inputSchema: Schema,
        fn: UserFunction,
        targetCacheId: str = None,
        shouldProfile=False,
    ):
        super().__init__(
            inputSchema=inputSchema,
            outputSchema=fn.outputSchema,
            shouldProfile=shouldProfile,
        )
        self.inputSchema = inputSchema
        self.fn = fn
        self.targetCacheId = targetCacheId
        if not inputSchema == fn.inputSchema:
            raise Exception(
                "Supplied UserFunction input schema does not match input schema"
            )

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, ApplyUserFunctionOp)
            and self.fn == other.fn
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            "ApplyUserFunctionOp("
            + str(self.outputSchema)
            + ", "
            + "Function: "
            + str(self.fn.udfid)
            + ")"
        )

    def copy(self):
        return ApplyUserFunctionOp(
            inputSchema=self.inputSchema,
            fn=self.fn,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def opId(self):
        raise NotImplementedError("Legacy method")
        d = {
            "operator": "ApplyUserFunctionOp",
            "source": self.source.opId(),
            "fn": str(self.fn.udfid),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_estimate_sample_data: List[Dict[str, Any]] = None):
        # get input estimates and pass through to output
        inputEstimates, subPlanCostEst = self.source.estimateCost(
            cost_estimate_sample_data
        )
        outputEstimates = {**inputEstimates}

        # the profiler will record selectivity and timing info for this operator,
        # which can be used to improve timing related estimates
        if cost_estimate_sample_data is not None:
            # compute estimates
            filter = f"(filter == '{str(self.filter)}') & (op_name == 'p_filter')"
            time_per_record = StatsProcessor._est_time_per_record(
                cost_estimate_sample_data, filter=filter
            )
            selectivity = StatsProcessor._est_selectivity(
                cost_estimate_sample_data, filter=filter, model_name=self.model.value
            )

            # estimate cardinality using sample selectivity and input cardinality est.
            cardinality = inputEstimates["cardinality"] * selectivity

            # update cardinality, timePerElement and related stats
            outputEstimates["cardinality"] = cardinality
            outputEstimates["timePerElement"] = time_per_record
            outputEstimates["cumulativeTimePerElement"] = (
                inputEstimates["cumulativeTimePerElement"] + time_per_record
            )
            outputEstimates["totalTime"] = (
                cardinality * time_per_record + inputEstimates["totalTime"]
            )

            return outputEstimates, {
                "cumulative": outputEstimates,
                "thisPlan": {
                    "time_per_record": time_per_record,
                    "selectivity": selectivity,
                },
                "subPlan": subPlanCostEst,
            }

        # for now, assume applying the user function takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates, {
            "cumulative": outputEstimates,
            "thisPlan": {},
            "subPlan": subPlanCostEst,
        }

    def __iter__(self):
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)

        @self.profile(name="applyfn", shouldProfile=self.shouldProfile)
        def iteratorFn():
            for nextCandidate in self.source:
                try:
                    dr = self.fn.map(nextCandidate)
                    if shouldCache:
                        datadir.appendCache(self.targetCacheId, dr)
                    yield dr
                except Exception as e:
                    print("Error in applying function", e)
                    pass

            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()


class ApplyAverageAggregateOp(PhysicalOp):
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, ApplyAverageAggregateOp)
            and self.aggFunction == other.aggFunction
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            "ApplyAverageAggregateOp("
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

    def opId(self):
        raise NotImplementedError("Legacy method")
        d = {
            "operator": "ApplyAverageAggregateOp",
            "source": self.source.opId(),
            "aggFunction": str(self.aggFunction),
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        # get input estimates and pass through to output
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record timing info for this operator, which can be used
        # to improve timing related estimates
        op_filter = "(op_name == 'average')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # compute estimates
            time_per_record = cost_est_data[op_filter]["time_per_record"]

            # output cardinality for an aggregate will be 1
            cardinality = 1

            # update cardinality, timePerElement and related stats
            outputEstimates["cardinality"] = cardinality
            outputEstimates["timePerElement"] = time_per_record
            outputEstimates["cumulativeTimePerElement"] = (
                inputEstimates["cumulativeTimePerElement"] + time_per_record
            )
            outputEstimates["totalTime"] = (
                cardinality * time_per_record + inputEstimates["totalTime"]
            )

            return outputEstimates, {
                "cumulative": outputEstimates,
                "thisPlan": {"time_per_record": time_per_record},
                "subPlan": subPlanCostEst,
            }

        # output cardinality for an aggregate will be 1
        outputEstimates["cardinality"] = 1

        # for now, assume applying the aggregate takes negligible additional time (and no cost in USD)
        outputEstimates["timePerElement"] = 0
        outputEstimates["usdPerElement"] = 0
        outputEstimates["estOutputTokensPerElement"] = 0

        return outputEstimates, {
            "cumulative": outputEstimates,
            "thisPlan": {},
            "subPlan": subPlanCostEst,
        }

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


class LimitScanOp(PhysicalOp):
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

        # NOTE: need to construct profiler after all fields used by self.opId() are set
        self.profiler = Profiler(op_id=self.opId())
        self.profile = self.profiler.iter_profiler

    def __eq__(self, other: PhysicalOp):
        return (
            isinstance(other, LimitScanOp)
            and self.limit == other.limit
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
        )

    def __str__(self):
        return (
            "LimitScanOp("
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

    def opId(self):
        raise NotImplementedError("Legacy method")
        d = {
            "operator": "LimitScanOp",
            "outputSchema": str(self.outputSchema),
            "source": self.source.opId(),
            "limit": self.limit,
            "targetCacheId": self.targetCacheId,
        }
        ordered = json.dumps(d, sort_keys=True)
        return hashlib.sha256(ordered.encode()).hexdigest()[:MAX_ID_CHARS]

    def estimateCost(self, cost_est_data: Dict[str, Any] = None):
        # get input estimates and pass through to output
        inputEstimates, subPlanCostEst = self.source.estimateCost(cost_est_data)
        outputEstimates = {**inputEstimates}

        # the profiler will record selectivity and timing info for this operator,
        # which can be used to improve timing related estimates
        op_filter = "(op_name == 'limit')"
        if cost_est_data is not None and cost_est_data[op_filter] is not None:
            # compute estimates
            time_per_record = cost_est_data[op_filter]["time_per_record"]

            # output cardinality for limit can be at most self.limit
            cardinality = min(self.limit, inputEstimates["cardinality"])

            # update cardinality, timePerElement and related stats
            outputEstimates["cardinality"] = cardinality
            outputEstimates["timePerElement"] = time_per_record
            outputEstimates["cumulativeTimePerElement"] = (
                inputEstimates["cumulativeTimePerElement"] + time_per_record
            )
            outputEstimates["totalTime"] = (
                cardinality * time_per_record + inputEstimates["totalTime"]
            )

            return outputEstimates, {
                "cumulative": outputEstimates,
                "thisPlan": {"time_per_record": time_per_record},
                "subPlan": subPlanCostEst,
            }

        # output cardinality for limit can be at most self.limit
        outputEstimates["cardinality"] = min(self.limit, inputEstimates["cardinality"])

        return outputEstimates, {
            "cumulative": outputEstimates,
            "thisPlan": {},
            "subPlan": subPlanCostEst,
        }

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
