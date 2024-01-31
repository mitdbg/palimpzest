from palimpzest.elements import *
from palimpzest.solver import Solver

class PhysicalOp:
    synthesizedFns = {}
    solver = Solver()

    def __init__(self, outputElementType):
        self.outputElementType = outputElementType

    def getNext(self):
        raise NotImplementedError("Abstract method")
    
    def finalize(self, datasource):
        raise NotImplementedError("Abstract method")
    
    def dumpPhysicalTree(self):
        raise NotImplementedError("Abstract method")

class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputElementType):
        super().__init__(outputElementType=outputElementType)
        self.datasource = None

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputElementType) + ")"

    def finalize(self, datasource):
        self.datasource = datasource

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, None)

    def __iter__(self):
        if self.datasource is None:
            raise Exception("InduceFromCandidateOp has not been finalized with a datasource")

        def iteratorFn():    
            for nextCandidate in self.datasource:
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
    def __init__(self, outputElementType, source, filters):
        super().__init__(outputElementType=outputElementType)
        self.source = source
        self.filters = filters

    def __str__(self):
        filterStr = "and ".join([str(f) for f in self.filters])
        return "FilterCandidateOp(" + str(self.outputElementType) + ", " + "Filters: " + str(filterStr) + ")"

    def finalize(self, datasource):
        self.source.finalize(datasource)

    def dumpPhysicalTree(self):
        """Return the physical tree of operators."""
        return (self, self.source.dumpPhysicalTree())

    def __iter__(self):
        def iteratorFn():
            for nextCandidate in self.source: 
                if self._passesFilters(nextCandidate):
                    yield nextCandidate
        return iteratorFn()

    def _passesFilters(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        taskDescriptor = ("FilterCandidateOp", self.filters, candidate.element, self.outputElementType)
        if not taskDescriptor in PhysicalOp.synthesizedFns:
            PhysicalOp.synthesizedFns[taskDescriptor] = PhysicalOp.solver.synthesize(taskDescriptor)
        return PhysicalOp.synthesizedFns[taskDescriptor](candidate)
