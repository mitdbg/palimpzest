import os

from palimpzest import Field
from palimpzest.elements import DataRecord, TextFile, File, PDFFile, ImageFile, Number
from palimpzest.tools import cosmos_client, get_text_from_pdf, processPapermagePdf
from palimpzest.tools.dspysearch import run_cot_bool, run_cot_qa, gen_filter_signature_class, gen_qa_signature_class
from palimpzest.datasources import DataDirectory
from palimpzest.tools.openai_image_converter import do_image_analysis
import json
from papermage import Document
import base64
import modal

class Solver:
    """This solves for needed operator implementations"""
    def __init__(self, verbose = False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((ImageFile, File))
        self._hardcodedFns.add((TextFile, File))
        self._verbose = verbose

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

    def _makeHardCodedTypeConversionFn(self, outputElement, inputElement, config):
        """This converts from one type to another when we have a hard-coded method for doing so."""
        if outputElement == PDFFile and inputElement == File:
            if config.get("pdfprocessing") == "modal":
                print("handling PDF processing remotely")
                remoteFunc = modal.Function.lookup("palimpzest.tools", "processPapermagePdf")
            else:
                remoteFunc = None
                
            def _fileToPDF(candidate: DataRecord):
                pdf_bytes = candidate.contents
                pdf_filename = candidate.filename
                if remoteFunc is not None:
                    docJsonStr = remoteFunc.remote([pdf_bytes])
                    docdict = json.loads(docJsonStr[0])
                    doc = Document.from_json(docdict)
                    text_content = ""
                    for p in doc.pages:
                        text_content += p.text
                else:
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
                text_content = str(candidate.contents, 'utf-8')
                dr = DataRecord(outputElement)
                dr.filename = candidate.filename
                dr.contents = text_content
                return dr
            return _fileToText
        elif outputElement == ImageFile and inputElement == File:
            def _fileToImage(candidate: DataRecord):
                if not candidate.element == inputElement:
                    return None
                # b64 decode of candidate.contents
                #print(candidate.contents)
                image_bytes = base64.b64encode(candidate.contents).decode('utf-8')
                #image_bytes = candidate.contents #base64.b64decode(candidate.contents)#.decode("utf-8")
                dr = DataRecord(outputElement)
                dr.filename = candidate.filename
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                # get openai key from environment
                openai_key = os.environ['OPENAI_API_KEY']
                dr.contents = do_image_analysis(openai_key, image_bytes)
                return dr
            return _fileToImage

        else:
            raise Exception(f"Cannot hard-code conversion from {inputElement} to {outputElement}")

    def _makeLLMTypeConversionFn(self, outputElement, inputElement, config, conversionDesc):
            llmservice = config.get("llmservice", "openai")
            def fn(candidate: DataRecord):
                # iterate through all empty fields in the outputElement and ask questions to fill them
                # for field in inputElement.__dict__:
                dr = DataRecord(outputElement)
                text_content = candidate.asTextJSON()
                doc_schema = str(outputElement)
                doc_type = outputElement.className()

                for field_name in outputElement.fieldNames():
                    f = getattr(outputElement, field_name)
                    try:
                        answer = run_cot_qa(text_content, 
                                            f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.",
                                            llmService=llmservice, verbose=self._verbose, promptSignature=gen_qa_signature_class(doc_schema, doc_type))
                        setattr(dr, field_name, answer)
                    except Exception as e:
                        print(f"Error: {e}")
                        setattr(dr, field_name, None)
                return dr
            return fn

    def _makeFilterFn(self, taskDescriptor, config):
            # parse inputs
            _, functionParams, _, inputElement = taskDescriptor
            llmservice = config.get("llmservice", "openai")
            doc_schema = str(inputElement)
            doc_type = inputElement.className()
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
                    response = run_cot_bool(text_content, filterCondition, llmService=llmservice,
                                               verbose=self._verbose, promptSignature=gen_filter_signature_class(doc_schema, doc_type))
                    if response == "TRUE":
                        return True
                    else:
                        return False
                return llmFilter
            return createLLMFilter("and ".join([str(f) for f in functionParams]))

    def synthesize(self, taskDescriptor, config):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputElement, inputElement = taskDescriptor

        if functionName == "InduceFromCandidateOp" or functionName == "ParallelInduceFromCandidateOp":
            conversionDesc = functionParams
            typeConversionDescriptor = (outputElement, inputElement)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(outputElement, inputElement)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(outputElement, inputElement, config)
            else:
                return self._makeLLMTypeConversionFn(outputElement, inputElement, config, conversionDesc)
        elif functionName == "FilterCandidateOp" or functionName == "ParallelFilterCandidateOp":
            return  self._makeFilterFn(taskDescriptor, config)
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
