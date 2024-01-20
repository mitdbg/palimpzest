from palimpzest.elements import *
from .loaders import *

class Processor():
    """A processor applies an extractor to a particular input"""
    def __init__(self, rootElement=None, populatedElements=None, exampleRepos=None, streaming=False):
        self.rootElement = rootElement
        self.populatedElements = {key: value for key, value in populatedElements}
        self.exampleRepos = exampleRepos
        self.streaming = streaming

    def compile(self):
        """Compile the given tree of elements into a runnable program"""

        def countSteps(element, idx=0):
            #print(" " * idx, f"{idx} countSteps({element})")
            return 1 + sum(countSteps(x, idx=idx+1) for x in element.children)

        def countDepth(element, idx=0):
            #print(" " * idx, f"{idx} countDepth({element})")
            return 1 + max([0] + list(countDepth(x, idx=idx+1) for x in element.children))

        return {"steps": countSteps(self.rootElement),
                "depth": countDepth(self.rootElement),
                "rootElement": self.rootElement,
                "leafSources": self.populatedElements,
                "exampleRepos": self.exampleRepos,
                "streaming": self.streaming}

    def dumpLogicalTree(self):
        """Dump the logical tree of elements"""
        def dumpElement(nodeTuple, idx=0):
            label, nodeElt, childNodes = nodeTuple
            print(" " * idx, f"{idx} {label}")
            if nodeElt in self.populatedElements:
                print(" " * idx, f"  {nodeElt} <-- populated by {self.populatedElements[nodeElt]}")
            for x in childNodes:
                dumpElement(x, idx=idx+1)
        dumpElement(self.rootElement.getLogicalTree())

    def execute(self):
        pass
