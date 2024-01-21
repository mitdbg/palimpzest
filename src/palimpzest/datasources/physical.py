from palimpzest.elements import *

class PhysicalOp:
    def __init__(self):
        pass


class PopulateOp(PhysicalOp):
    def __init__(self, element, populatedSrcElement):
        self.element = element
        self.populatedSrcElement = populatedSrcElement

    def dump(self, idx=0, verbose=False):
        print(" " * idx, f"PopulateOp({self.element}, {self.populatedSrcElement})")
        if verbose:
            print(" " * idx, self.makeDataFreePrompt())

    def makeDataFreePrompt(self):
        prompt = ("You are mapping a data object from one representation to another. "
                  "You will be given an object called INPUT and be asked to generate an object called OUTPUT. "
                  "Both objects are in JSON format. \n"
                  f"INPUT can be described as: {self.populatedSrcElement.desc} \n"
                  f"OUTPUT can be described as: {self.element.desc} \n"
                  "Here is the INPUT object in JSON format: {}"
                  "Please emit the OUTPUT object in JSON format. Do not include any other text."
                    "If you cannot generate the OUTPUT object, please type 'skip'.")
        return prompt
    
class MapOp(PhysicalOp):
    def __init__(self, element, childOp):
        self.element = element
        self.childOp = childOp

    def dump(self, idx=0, verbose=False):
        print(" " * idx, f"MapOp({self.element}, <child>)")
        if verbose:
            print(" " * idx, self.makeDataFreePrompt())
        print()
        self.childOp.dump(idx=idx+1, verbose=verbose)

    def makeDataFreePrompt(self):
        prompt = ("You are mapping a data object from one representation to another. "
                  "You will be given an object called INPUT and be asked to generate an object called OUTPUT. "
                  "Both objects are in JSON format. \n"
                  f"INPUT can be described as: {self.childOp.element.desc} \n"
                  f"OUTPUT can be described as: {self.element.desc} \n"
                  "Here is the INPUT object in JSON format: {}"
                  "Please emit the OUTPUT object in JSON format. Do not include any other text."
                    "If you cannot generate the OUTPUT object, please type 'skip'.")
        return prompt

class AggregateOp(PhysicalOp):
    def __init__(self, element, childOps):
        self.element = element
        self.childOps = childOps

    def dump(self, idx=0, verbose=False):
        print(" " * idx, f"AggregateOp({self.element} [children])")
        if verbose:
            print(" " * idx, self.makeDataFreePrompt())
        print()
        for x in self.childOps:
            x.dump(idx=idx+1)

    def makeDataFreePrompt(self):
        prompt = ("You are created a new data object out of a set of smaller ones. This is similar to creating a record out of a set of fields. "
                  "You will be given a set of objects called INPUT-1, INPUT-2, INPUT-3, and so on. You will be asked to generate an object called OUTPUT. "
                  "All objects are in JSON format. \n")
        for idx, childOp in enumerate(self.childOps):
            prompt += (f"INPUT-{idx+1} can be described as: {childOp.element.desc} \n")
        
        prompt += f"OUTPUT can be described as: {self.element.desc} \n"
                   
        for idx, childOp in enumerate(self.childOps):
            prompt += ("Here is INPUT-{idx+1} in JSON format: {} \n")
        
        prompt += ("Please emit the OUTPUT object in JSON format. Do not include any other text."
                    "If you cannot generate the OUTPUT object, please type 'skip'.")
        return prompt






