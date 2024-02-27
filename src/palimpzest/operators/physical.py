from palimpzest.constants import *
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import *
from palimpzest.solver import Solver

from __future__ import annotations
from typing import Any, Dict, Tuple, Union

import concurrent


class PhysicalOp:
    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    synthesizedFns = {}
    solver = Solver(verbose=LOG_LLM_OUTPUT)

    def __init__(self, outputSchema: Schema) -> None:
        self.outputSchema = outputSchema
        self.datadir = DataDirectory()

    def dumpPhysicalTree(self) -> Tuple[PhysicalOp, Union[PhysicalOp, None]]:
        raise NotImplementedError("Abstract method")

    def estimateCost(self) -> Dict[str, Any]:
        """Returns dict of (cardinality, timePerElement, costPerElement, startupTime, startupCost)"""
        raise NotImplementedError("Abstract method")

class MarshalAndScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, concreteDatasetIdentifier: str):
        super().__init__(outputSchema=outputSchema)
        self.concreteDatasetIdentifier = concreteDatasetIdentifier

    def __str__(self):
        return "MarshalAndScanDataOp(" + str(self.outputSchema) + ", " + self.concreteDatasetIdentifier + ")"
    
    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)
    
    def estimateCost(self):
        cardinality = self.datadir.getCardinality(self.concreteDatasetIdentifier) + 1
        size = self.datadir.getSize(self.concreteDatasetIdentifier)
        perElementSizeInKb = (size / float(cardinality)) / 1024.0
        timePerElement = LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
        costPerElement = 0
        startupTime = 0
        startupCost = 0

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": size,
            "bytesReadRemotely": 0
        }
    
    def __iter__(self):
        def iteratorFn():
            for nextCandidate in self.datadir.getRegisteredDataset(self.concreteDatasetIdentifier):
                yield nextCandidate
        return iteratorFn()

class CacheScanDataOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, cacheIdentifier: str):
        super().__init__(outputSchema=outputSchema)
        self.cacheIdentifier = cacheIdentifier

    def __str__(self):
        return "CacheScanDataOp(" + str(self.outputSchema) + ", " + self.cacheIdentifier + ")"
    
    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)

    def estimateCost(self):
        cardinality = sum(1 for _ in self.datadir.getCachedResult(self.cacheIdentifier)) + 1
        # TODO: use something similar to datadir.getSize() to compute this
        size = 100 * cardinality
        perElementSizeInKb = (size / float(cardinality)) / 1024.0
        timePerElement = LOCAL_SCAN_TIME_PER_KB * perElementSizeInKb
        costPerElement = 0
        startupTime = 0
        startupCost = 0

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": size,
            "bytesReadRemotely": 0
        }

    def __iter__(self):
        def iteratorFn():
            for nextCandidate in self.datadir.getCachedResult(self.cacheIdentifier):
                yield nextCandidate
        return iteratorFn()


class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.targetCacheId = targetCacheId

        taskDescriptor = ("InduceFromCandidateOp", None, outputSchema, source.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputSchema) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

        # TODO: convert LOCAL_LLM_CONVERSION_TIME_PER_RECORD into a model-specific time estimate
        # TODO: can selectivity be greater than one? i.e., can the induce operation produce multiple outputs per file?
        # TODO: costPerElement needs to be computed as a fcn. of the LLM being used, number of tokens for input, est. num tokens for output.
        selectivity = 1.0
        cardinality = selectivity * inputCostEstimates["cardinality"]
        timePerElement = LOCAL_LLM_CONVERSION_TIME_PER_RECORD + inputCostEstimates["timePerElement"]
        costPerElement = inputCostEstimates["costPerElement"]
        startupTime = inputCostEstimates["startupTime"]
        startupCost = inputCostEstimates["startupCost"]
        bytesReadLocally = inputCostEstimates["bytesReadLocally"]
        bytesReadRemotely = inputCostEstimates["bytesReadRemotely"]

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": bytesReadLocally,
            "bytesReadRemotely": bytesReadRemotely
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)
        def iteratorFn():    
            for nextCandidate in self.source:
                resultRecord = self._attemptMapping(nextCandidate)
                if resultRecord is not None:
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, resultRecord)
                    yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = ("InduceFromCandidateOp", None, self.outputSchema, candidate.schema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelInduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.targetCacheId = targetCacheId

        taskDescriptor = ("ParallelInduceFromCandidateOp", None, outputSchema, source.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "ParallelInduceFromCandidateOp(" + str(self.outputSchema) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

        # TODO: same questions as for non-parallel induce; Also need to figure out where to compute timePerElement * numElts / parallelism for total latency est.
        selectivity = 1.0
        cardinality = selectivity * inputCostEstimates["cardinality"]
        timePerElement = LOCAL_LLM_CONVERSION_TIME_PER_RECORD + inputCostEstimates["timePerElement"]
        costPerElement = inputCostEstimates["costPerElement"]
        startupTime = inputCostEstimates["startupTime"]
        startupCost = inputCostEstimates["startupCost"]
        bytesReadLocally = inputCostEstimates["bytesReadLocally"]
        bytesReadRemotely = inputCostEstimates["bytesReadRemotely"]

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": bytesReadLocally,
            "bytesReadRemotely": bytesReadRemotely
        }

    def __iter__(self):
        # This is very crudely implemented right now, since we materialize everything
        shouldCache = self.datadir.openCache(self.targetCacheId)
        def iteratorFn():
            chunksize = 20 + 2
            inputs = []
            results = []

            for nextCandidate in self.source:
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks of size chunkSize
            with concurrent.futures.ThreadPoolExecutor(max_workers=chunksize) as executor:
                results = list(executor.map(self._attemptMapping, inputs, chunksize=chunksize))

                for resultRecord in results:
                    if resultRecord is not None:
                        if shouldCache:
                            self.datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputSchema. Return None if it fails."""
        taskDescriptor = ("ParallelInduceFromCandidateOp", None, self.outputSchema, candidate.schema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class FilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.filter = filter
        self.targetCacheId = targetCacheId

        taskDescriptor = ("FilterCandidateOp", (self.filter,), source.outputSchema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "FilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

        # TODO: need to estimate selectivity somehow
        # TODO: convert LOCAL_LLM_CONVERSION_TIME_PER_RECORD into a model-specific time estimate
        # TODO: costPerElement needs to be computed as a fcn. of the LLM being used, number of tokens for input, est. num tokens for output.
        selectivity = 1.0
        cardinality = selectivity * inputCostEstimates["cardinality"]
        timePerElement = LOCAL_LLM_FILTER_TIME_PER_RECORD + inputCostEstimates["timePerElement"]
        costPerElement = inputCostEstimates["costPerElement"]
        startupTime = inputCostEstimates["startupTime"]
        startupCost = inputCostEstimates["startupCost"]
        bytesReadLocally = inputCostEstimates["bytesReadLocally"]
        bytesReadRemotely = inputCostEstimates["bytesReadRemotely"]

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": bytesReadLocally,
            "bytesReadRemotely": bytesReadRemotely
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)
        def iteratorFn():
            for nextCandidate in self.source: 
                if self._passesFilter(nextCandidate):
                    if shouldCache:
                        self.datadir.appendCache(self.targetCacheId, nextCandidate)
                    yield nextCandidate
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("FilterCandidateOp", (self.filter,), candidate.schema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelFilterCandidateOp(PhysicalOp):
    def __init__(self, outputSchema: Schema, source: PhysicalOp, filter: Filter, targetCacheId: str=None):
        super().__init__(outputSchema=outputSchema)
        self.source = source
        self.filter = filter
        self.targetCacheId = targetCacheId

        taskDescriptor = ("ParallelFilterCandidateOp", (self.filter,), source.outputSchema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = self.datadir.current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "ParallelFilterCandidateOp(" + str(self.outputSchema) + ", " + "Filter: " + str(self.filter) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

        # TODO: same questions as for non-parallel induce; Also need to figure out where to compute timePerElement * numElts / parallelism for total latency est.
        selectivity = 1.0
        cardinality = selectivity * inputCostEstimates["cardinality"]
        timePerElement = LOCAL_LLM_FILTER_TIME_PER_RECORD + inputCostEstimates["timePerElement"]
        costPerElement = inputCostEstimates["costPerElement"]
        startupTime = inputCostEstimates["startupTime"]
        startupCost = inputCostEstimates["startupCost"]
        bytesReadLocally = inputCostEstimates["bytesReadLocally"]
        bytesReadRemotely = inputCostEstimates["bytesReadRemotely"]

        return {
            "cardinality": cardinality,
            "timePerElement": timePerElement,
            "costPerElement": costPerElement,
            "startupTime": startupTime,
            "startupCost": startupCost,
            "bytesReadLocally": bytesReadLocally,
            "bytesReadRemotely": bytesReadRemotely
        }

    def __iter__(self):
        shouldCache = self.datadir.openCache(self.targetCacheId)
        def iteratorFn():
            chunksize = 20 + 2
            inputs = []
            results = []

            for nextCandidate in self.source: 
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks of size chunkSize
            with concurrent.futures.ThreadPoolExecutor(max_workers=chunksize) as executor:
                results = list(executor.map(self._passesFilter, inputs, chunksize=chunksize))

                for idx, filterResult in enumerate(results):
                    if filterResult:
                        resultRecord = inputs[idx]
                        if shouldCache:
                            self.datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord
            if shouldCache:
                self.datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilter(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("ParallelFilterCandidateOp", (self.filter,), candidate.schema, self.outputSchema)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)

        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)
