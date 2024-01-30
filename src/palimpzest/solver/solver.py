from palimpzest.elements import DataRecord

class Solver:
    """This solves for needed operator implementations"""

    def __init__(self):
        pass

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, outputElement, inputElement = taskDescriptor

        if functionName == "InduceFromCandidateOp":
            def fn(candidate: DataRecord):
                if candidate.element == inputElement:
                    dr = DataRecord(outputElement)
                    dr.title = "Test Title"
                    dr.contents = candidate.contents
                    dr.filename = candidate.filename
                    return dr
                else:
                    return None
            return fn
        elif functionName == "FilterCandidateOp":
            def fn(candidate: DataRecord):
                if candidate.element == inputElement:
                    return True
                else:
                    return False
            return fn
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
        