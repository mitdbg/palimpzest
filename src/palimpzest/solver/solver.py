import os

from palimpzest import Field
from palimpzest.elements import DataRecord, TextFile, File, PDFFile
from palimpzest.tools import cosmos_client, get_text_from_pdf
from palimpzest.tools.dspysearch import run_rag_boolean, run_rag_qa

class Solver:
    """This solves for needed operator implementations"""
    def __init__(self):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._simpleTypeConversions.add((TextFile, File))
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))

    def easyConversionAvailable(self, outputElement, inputElement):
        return (outputElement, inputElement) in self._simpleTypeConversions or (outputElement, inputElement) in self._hardcodedFns

    def _makeSimpleTypeConversionFn(self, outputElement, inputElement):
        """This is a very simple function that converts a DataRecord from one type to another, when we know they have identical fields."""
        def _simpleTypeConversionFn(candidate: DataRecord):
            if not candidate.element == inputElement:
                return None
            
            dr = DataRecord(outputElement)
            for field in outputElement.fieldNames():
                if hasattr(candidate, field):
                    setattr(dr, field, getattr(candidate, field))
                elif field.required:
                    return None
            return dr
        return _simpleTypeConversionFn

    def _makeHardCodedTypeConversionFn(self, outputElement, inputElement):
        """This converts from one type to another when we have a hard-coded method for doing so."""
        if outputElement == PDFFile and inputElement == File:
            def _fileToPDF(candidate: DataRecord):
                if not candidate.element == inputElement:
                    return None
                pdf_bytes = candidate.contents
                pdf_filename = candidate.filename
                print("About to process PDF for ", pdf_filename)
                text_content = get_text_from_pdf(candidate.filename, candidate.contents)
                dr = DataRecord(outputElement)
                dr.filename = pdf_filename
                dr.contents = pdf_bytes
                dr.text_contents = text_content
                return dr
            return _fileToPDF
        else:
            raise Exception(f"Cannot hard-code conversion from {inputElement} to {outputElement}")

    def _makeLLMTypeConversionFn(self, outputElement, inputElement):
            def fn(candidate: DataRecord):
                # iterate through all empty fields in the outputElement and ask questions to fill them
                # for field in inputElement.__dict__:
                dr = DataRecord(outputElement)
                text_content = candidate.asJSON()
                for field_name in outputElement.fieldNames():
                    f = getattr(outputElement, field_name)
                    answer = run_rag_qa(text_content, f"What is the {field_name} of the document? ({f.desc})")
                    setattr(dr, field_name, answer)
                return dr
            return fn

    def _makeFilterFn(self, taskDescriptor):
            functionName, functionParams, outputElement, inputElement = taskDescriptor
            if len(functionParams) == 0:
                def allPass(candidate: DataRecord):
                    return True

                return allPass
            
            # By default, a filter requires an LLM invocation to run
            # Someday maybe we will offer the user the chance to run a hard-coded function.
            # Or maybe we will ask the LLM to synthesize traditional code here.
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    if not candidate.element == inputElement:
                        return False
                    
                    text_content = candidate.asJSON()
                    response = run_rag_boolean(text_content, filterCondition)
                    if response == "TRUE":
                        return True
                    else:
                        return False
                return llmFilter
            return createLLMFilter("and ".join([str(f) for f in functionParams]))

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputElement, inputElement = taskDescriptor

        if functionName == "InduceFromCandidateOp":
            typeConversionDescriptor = (outputElement, inputElement)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(outputElement, inputElement)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(outputElement, inputElement)
            else:
                return self._makeLLMTypeConversionFn(outputElement, inputElement)
        elif functionName == "FilterCandidateOp":
            return  self._makeFilterFn(taskDescriptor)
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
