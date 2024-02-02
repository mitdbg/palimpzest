import os

from palimpzest.elements import DataRecord
from palimpzest.tools import cosmos_client
from palimpzest.tools.dspysearch import run_rag


class Solver:
    """This solves for needed operator implementations"""

    def __init__(self):
        self._hardcodedFns = {}

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputElement, inputElement = taskDescriptor
        # print(f"Synthesizing function for task: {functionName} with params {functionParams} from {inputElement} to {outputElement}")
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
        # Chunwei: I will try to implement a few function templates here. Then serve them to LLM models for code generation for similar tasks.
        # Chunwei: The generated functions will be cached in the synthesizedFns dictionary and profiled (runtime, resource usage) everytime they are called.
        #          The profiling results will be used to rank the functions and choose the best one based on the hardware constraints.
        ######################################
        if functionName == "InduceFromCandidateOp":

            # Let's check if there's a prefab function for this mapping from type A to type B
            # hardcodedFn = self._hardcodedFn(inputElement, outputElement)
            # if hardcodedFn is not None:
            #    return hardcodedFn

            # Let's do LLM-based induction by default
            # REMIND: Chunwei, let's do some LLM-based induction here
            def fn(candidate: DataRecord):
                print(f"file name: {candidate.filename}")
                if candidate.element is inputElement:
                    print(f"filtered file name: {candidate.filename}")
                    dr = DataRecord(outputElement)
                    dr.title = "Test Title"
                    dr.contents = candidate.contents
                    dr.filename = candidate.filename
                    return dr
                else:
                    return None

            return fn

        elif functionName == "FilterCandidateOp":
            if len(functionParams) == 0:
                def allPass(candidate: DataRecord):
                    if candidate.element == inputElement:
                        return True
                    else:
                        return False
                return allPass



            # Let's do LLM-based filters by default
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    if candidate.element == inputElement:
                        pdf_bytes = candidate.contents
                        pdf_filename = candidate.filename
                        file_name = os.path.basename(pdf_filename)
                        cosmos_file_dir = file_name.split('.')[0].replace(' ', '_')
                        output_dir = os.path.dirname(pdf_filename)
                        print(f"Processing {file_name}")
                        # Call the cosmos_client function
                        cosmos_client(file_name, pdf_bytes, output_dir)
                        text_path = os.path.join(output_dir, f"{cosmos_file_dir}/{file_name.split('.')[0]}.txt")
                        print(f"Text file path: {text_path}")
                        with open(text_path, 'r') as file:
                            text_content = file.read()
                        response = run_rag(text_content, filterCondition)
                        if response == "TRUE":
                            return True
                    return False
                return llmFilter
            return createLLMFilter("and ".join([str(f) for f in functionParams]))
            # def createLLMFilter(filterCondition: str):
            #    def llmFilter(candidate: DataRecord):
            #        if candidate.element == inputElement:
            #            prompt = "Below is a filter condition in natural language called FILTER and a data record " +
            #                     "called RECORD. Please return just one of two values: TRUE or FALSE. Return TRUE if " + 
            #                     "FILTER accurately describes the RECORD. Return FALSE otherwise.\n\n" +
            #                     f"FILTER: {filterCondition}\n\n" +
            #                     f"RECORD: {candidate.contents}"
            #            response = openAILLMThing.prompt(prompt)
            #            if response == "TRUE":
            #                return True        
            #        return False
            # return createLLMFilter("and ".join(functionParams[0]))
            # def fn(candidate: DataRecord):
            #     if candidate.element == inputElement:
            #         return True
            #     else:
            #         return False
            # return fn
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
