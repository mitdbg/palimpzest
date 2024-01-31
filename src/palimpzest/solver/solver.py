from palimpzest.elements import DataRecord

class Solver:
    """This solves for needed operator implementations"""

    def __init__(self):
        self._hardcodedFns = {}

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputElement, inputElement = taskDescriptor

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

            # Let's check if there's a prefab function for this mapping from type A to type B
            hardcodedFn = self._hardcodedFn(inputElement, outputElement)
            if hardcodedFn is not None:
                return hardcodedFn
            
            # Let's do LLM-based induction by default
            # REMIND: Chunwei, let's do some LLM-based induction here
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
            # Let's do LLM-based filters by default
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    if candidate.element == inputElement:
                        prompt = "Below is a filter condition in natural language called FILTER and a data record " +
                                 "called RECORD. Please return just one of two values: TRUE or FALSE. Return TRUE if " + 
                                 "FILTER accurately describes the RECORD. Return FALSE otherwise.\n\n" +
                                 f"FILTER: {filterCondition}\n\n" +
                                 f"RECORD: {candidate.contents}"
                        response = openAILLMThing.prompt(prompt)
                        if response == "TRUE":
                            return True        
                    return False
            return createLLMFilter("and ".join(functionParams[0]))
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
        