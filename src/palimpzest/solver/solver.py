from palimpzest.elements import DataRecord

class Solver:
    """This solves for needed operator implementations"""

    def __init__(self):
        pass

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, outputElement, inputElement = taskDescriptor

        ######################################
        #
        # Here is where we would synthesize the function.
        # We need to:
        # 1) Formulate a parameterizable prompt that accomplishes the task below.
        # 2) Then creates a function that calls OpenAI with that prompt, and parameterizes it with the given candidate object
        # 3) And when the call to OpenAI comes back, the function returns a DataRecord with the schema of `outputElement`
        #
        # This style of function should be created upon the call to synthesize(), and returned.
        # It will be registered inside the PhysicalOp.synthesizedFns dictionary.
        # It might be called multiple times, with many different candidate objects.
        #
        # If this system is working well, the function will be chosen to run fast, because it will get called for each candidate.
        # However, for now we can just call OpenAI. It's not super fast, but it's fine. Pretend we are an RDBMS in 1979. The plans
        #  aren't always great.
        #
        # In the future, we will be clever and choose the fastest possible LLM that can still deliver on the desired task,
        #  quality-wise.
        ######################################
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
        