import os

from palimpzest import Field
from palimpzest.elements import DataRecord, TextFile, File, PDFFile, ImageFile
from palimpzest.tools import cosmos_client, get_text_from_pdf
from palimpzest.tools.dspysearch import run_rag_boolean, run_rag_qa
from palimpzest.datasources import DataDirectory
from palimpzest.tools.openai_image_converter import do_image_analysis
import base64

class Solver:
    """This solves for needed operator implementations"""
    def __init__(self, verbose = False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((ImageFile, File))
        self._verbose = verbose

    def easyConversionAvailable(self, outputElement, inputElement):
        return (outputElement, inputElement) in self._simpleTypeConversions or (outputElement, inputElement) in self._hardcodedFns

    def _llmservice(self):
        # TODO: temporarily converting this into a function call;
        #       if this is triggered in __init__ for new users that
        #       have not yet set up their config(s), then this will
        #       lead to a chain of fcn. calls that causes an exception
        #       to be thrown on `import palimpzest`.
        llmservice = DataDirectory().config.get("llmservice")
        if llmservice is None:
            llmservice = "openai"
            print("LLM service has not been configured. Defaulting to openai.")
        
        return llmservice

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
        elif outputElement == TextFile and inputElement == File:
            def _fileToText(candidate: DataRecord):
                if not candidate.element == inputElement:
                    return None
                # b64 decode of candidate.contents
                text_content = base64.b64decode(candidate.contents).decode("utf-8")
                dr = DataRecord(outputElement)
                dr.filename = candidate.filename
                dr.contents = text_content
                return dr
            return _fileToText
        elif outputElement == ImageFile and inputElement == File:
            def _imageToText(candidate: DataRecord):
                if not candidate.element == inputElement:
                    return None
                # b64 decode of candidate.contents
                #print(candidate.contents)
                image_bytes = candidate.contents #base64.b64decode(candidate.contents)#.decode("utf-8")
                dr = DataRecord(outputElement)
                dr.filename = candidate.filename
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                # get openai key from environment
                openai_key = os.environ['OPENAI_API_KEY']
                dr.contents = do_image_analysis(openai_key, image_bytes)
                return dr
            return _imageToText

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
                    answer = run_rag_qa(text_content, f"What is the {field_name} of the document? ({f.desc})", llmService=self._llmservice(),verbose=self._verbose)
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
                    response = run_rag_boolean(text_content, filterCondition, llmService=self._llmservice(), verbose=self._verbose)
                    if response == "TRUE":
                        return True
                    else:
                        return False
                return llmFilter
            return createLLMFilter("and ".join([str(f) for f in functionParams]))

    def synthesize(self, taskDescriptor):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputElement, inputElement = taskDescriptor

        if functionName == "InduceFromCandidateOp" or functionName == "ParallelInduceFromCandidateOp":
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
