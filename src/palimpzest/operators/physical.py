from palimpzest.elements import *
from palimpzest.solver import Solver
from palimpzest.datasources import DataDirectory

import concurrent

# Assume 500 MB/sec for local SSD scan time
LOCAL_SCAN_TIME_PER_KB = 1 / (float(500) * 1024)

# Assume 10s per record for local LLM object conversion
LOCAL_LLM_CONVERSION_TIME_PER_RECORD = 10

# Assume 5s per record for local LLM boolean filter
LOCAL_LLM_FILTER_TIME_PER_RECORD = 5

logLLMOutput = False

class PhysicalOp:
    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    synthesizedFns = {}
    solver = Solver(verbose=logLLMOutput)

    def __init__(self, outputElementType):
        self.outputElementType = outputElementType

    def getNext(self):
        raise NotImplementedError("Abstract method")
    
    def dumpPhysicalTree(self):
        raise NotImplementedError("Abstract method")
    
    def estimateCost(self):
        """Returns dict of (cardinality, timePerElement, costPerElement, startupTime, startupCost)"""
        raise NotImplementedError("Abstract method")

class MarshalAndScanDataOp(PhysicalOp):
    def __init__(self, outputElementType, concreteDatasetIdentifier):
        super().__init__(outputElementType=outputElementType)
        self.concreteDatasetIdentifier = concreteDatasetIdentifier

    def __str__(self):
        return "MarshalAndScanDataOp(" + str(self.outputElementType) + ", " + self.concreteDatasetIdentifier + ")"
    
    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)
    
    def estimateCost(self):
        cardinality = DataDirectory().getCardinality(self.concreteDatasetIdentifier) + 1
        size = DataDirectory().getSize(self.concreteDatasetIdentifier)
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
            for nextCandidate in DataDirectory().getRegisteredDataset(self.concreteDatasetIdentifier):
                yield nextCandidate
        return iteratorFn()

class CacheScanDataOp(PhysicalOp):
    def __init__(self, outputElementType, cacheIdentifier):
        super().__init__(outputElementType=outputElementType)
        self.cacheIdentifier = cacheIdentifier

    def __str__(self):
        return "CacheScanDataOp(" + str(self.outputElementType) + ", " + self.cacheIdentifier + ")"
    
    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)

    def estimateCost(self):
        cardinality = sum(1 for _ in DataDirectory().getCachedResult(self.cacheIdentifier)) + 1
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
            for nextCandidate in DataDirectory().getCachedResult(self.cacheIdentifier):
                yield nextCandidate
        return iteratorFn()


class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source, targetCacheId=None):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.targetCacheId = targetCacheId

        taskDescriptor = ("InduceFromCandidateOp", None, outputElementType, source.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = DataDirectory().current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputElementType) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

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
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
        def iteratorFn():    
            for nextCandidate in self.source:
                resultRecord = self._attemptMapping(nextCandidate, self.outputElementType)
                if resultRecord is not None:
                    if shouldCache:
                        datadir.appendCache(self.targetCacheId, resultRecord)
                    yield resultRecord
            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord, outputElementType):
        """Attempt to map the candidate to the outputElementType. Return None if it fails."""
        taskDescriptor = ("InduceFromCandidateOp", None, outputElementType, candidate.element)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelInduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source, targetCacheId=None):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.targetCacheId = targetCacheId

        taskDescriptor = ("ParallelInduceFromCandidateOp", None, outputElementType, source.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = DataDirectory().current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        return "ParallelInduceFromCandidateOp(" + str(self.outputElementType) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

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
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
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
                            datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord
            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _attemptMapping(self, candidate: DataRecord):
        """Attempt to map the candidate to the outputElementType. Return None if it fails."""
        taskDescriptor = ("ParallelInduceFromCandidateOp", None, self.outputElementType, candidate.element)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class FilterCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source, filters, targetCacheId=None):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.filters = filters
        self.targetCacheId = targetCacheId

        taskDescriptor = ("FilterCandidateOp", tuple(self.filters), source.outputElementType, self.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = DataDirectory().current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "FilterCandidateOp(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

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
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
        def iteratorFn():
            for nextCandidate in self.source: 
                if self._passesFilters(nextCandidate):
                    if shouldCache:
                        datadir.appendCache(self.targetCacheId, nextCandidate)
                    yield nextCandidate
            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilters(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("FilterCandidateOp", tuple(self.filters), candidate.element, self.outputElementType)
        #print("LOOKING FOR FUNCTION", taskDescriptor)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)


class ParallelFilterCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source, filters, targetCacheId=None):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.filters = filters
        self.targetCacheId = targetCacheId

        taskDescriptor = ("ParallelFilterCandidateOp", tuple(self.filters), source.outputElementType, self.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            config = DataDirectory().current_config
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor, config)

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "ParallelFilterCandidateOp(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def estimateCost(self):
        inputCostEstimates = self.source.estimateCost()

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
        datadir = DataDirectory()
        shouldCache = datadir.openCache(self.targetCacheId)
        def iteratorFn():
            chunksize = 20 + 2
            inputs = []
            results = []

            for nextCandidate in self.source: 
                inputs.append(nextCandidate)

            # Grab items from the list inputs in chunks of size chunkSize
            with concurrent.futures.ThreadPoolExecutor(max_workers=chunksize) as executor:
                results = list(executor.map(self._passesFilters, inputs, chunksize=chunksize))

                for idx, filterResult in enumerate(results):
                    if filterResult:
                        resultRecord = inputs[idx]
                        if shouldCache:
                            datadir.appendCache(self.targetCacheId, resultRecord)
                        yield resultRecord
            if shouldCache:
                datadir.closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilters(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("ParallelFilterCandidateOp", tuple(self.filters), candidate.element, self.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            raise Exception("This function should have been synthesized during init():", taskDescriptor)

        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)
