from .elements import *

class PhysicalOp:
    def __init__(self, outputElementType):
        self.outputElementType = outputElementType

    def getNext(self):
        raise NotImplementedError("Abstract method")
    
    def finalize(self, datasource):
        raise NotImplementedError("Abstract method")

class InduceFromCandidateOp(PhysicalOp):
    def __init__(self, outputElementType):
        super().__init__(outputElementType=outputElementType)
        self.datasource = None

    def __str__(self):
        return "InduceFromCandidateOp(" + str(self.outputElementType) + ")"

    def finalize(self, datasource):
        self.datasource = datasource

    def getNext(self):
        if self.datasource is None:
            raise Exception("InduceFromCandidateOp has not been finalized with a datasource")
        
        while True:
            nextCandidate = self.source.getNext()
            if nextCandidate is None:
                return None
            else:
                resultRecord = self._attemptMapping(nextCandidate, self.outputElementType)
                if resultRecord is not None:
                    return resultRecord
                
    def _attemptMapping(candidate, outputElementType):
        """Attempt to map the candidate to the outputElementType. Return None if it fails."""
        raise NotImplementedError("I haven't done it yet!")


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

    def getNext(self):
        while True:
            nextCandidate = self.source.getNext()
            if nextCandidate is None:
                return None
            else:
                if self._passesFilters(nextCandidate):
                    return nextCandidate

    def _passesFilters(self, candidate):
        """Return True if the candidate passes all filters, False otherwise."""
        raise NotImplementedError("I haven't done it yet!")
        