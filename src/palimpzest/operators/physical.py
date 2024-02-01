from palimpzest.elements import *
from palimpzest.solver import Solver
from palimpzest.datasources import DataDirectory

class PhysicalOp:
    synthesizedFns = {}
    solver = Solver()

    def __init__(self, outputElementType):
        self.outputElementType = outputElementType

    def getNext(self):
        raise NotImplementedError("Abstract method")
    
    def dumpPhysicalTree(self):
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
    
    def __iter__(self):
        def iteratorFn():
            for nextCandidate in DataDirectory().getCachedResult(self.cacheIdentifier):
                yield nextCandidate
        return iteratorFn()


class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source):
        super().__init__(outputElementType=outputElementType)
        self.source = source

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputElementType) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def __iter__(self):
        def iteratorFn():    
            for nextCandidate in self.source:
                resultRecord = self._attemptMapping(nextCandidate, self.outputElementType)
                if resultRecord is not None:
                    yield resultRecord
        return iteratorFn()
                    
    def _attemptMapping(self, candidate: DataRecord, outputElementType):
        """Attempt to map the candidate to the outputElementType. Return None if it fails."""
        taskDescriptor = ("InduceFromCandidateOp", None, outputElementType, candidate.element)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)

class FilterCandidateOp(PhysicalOp):
    def __init__(self, outputElementType, source, filters, targetCacheId=None):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.filters = filters
        self.targetCacheId = targetCacheId

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "FilterCandidateOp(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def __iter__(self):
        shouldCache = DataDirectory().openCache(self.targetCacheId)
        def iteratorFn():
            for nextCandidate in self.source: 
                if self._passesFilters(nextCandidate):
                    if shouldCache:
                        DataDirectory().appendCache(self.targetCacheId, nextCandidate)
                    yield nextCandidate
            DataDirectory().closeCache(self.targetCacheId)

        return iteratorFn()

    def _passesFilters(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("FilterCandidateOp", tuple(self.filters), candidate.element, self.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)
