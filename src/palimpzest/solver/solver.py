from palimpzest.constants import Model, PromptStrategy
from palimpzest.elements import DataRecord, EquationImage, File, Filter, ImageFile, PDFFile, Schema, TextFile
from palimpzest.tools import get_text_from_pdf
from palimpzest.tools.dspysearch import run_cot_bool, run_cot_qa, gen_filter_signature_class, gen_qa_signature_class
from palimpzest.tools.openai_image_converter import do_image_analysis

from papermage import Document
from typing import Any, Dict, Tuple, Union

import json
import base64
import modal
import os

# DEFINITIONS
# TaskDescriptor = Tuple[str, Union[tuple, None], Schema, Schema]


from palimpzest.tools.skema_tools import equations_to_latex_base64, equations_to_latex


class Solver:
    """This solves for needed operator implementations"""
    def __init__(self, verbose: bool=False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((ImageFile, File))
        self._hardcodedFns.add((TextFile, File))
        # self._hardcodedFns.add((EquationImage, ImageFile))
        self._verbose = verbose

    def easyConversionAvailable(self, outputSchema: Schema, inputSchema: Schema):
        return (outputSchema, inputSchema) in self._simpleTypeConversions or (outputSchema, inputSchema) in self._hardcodedFns

    def _makeSimpleTypeConversionFn(self, outputSchema, inputSchema):
        """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""
        def _simpleTypeConversionFn(candidate: DataRecord):
            if not candidate.schema == inputSchema:
                return None

            dr = DataRecord(outputSchema)
            for field in outputSchema.fieldNames():
                if hasattr(candidate, field):
                    setattr(dr, field, getattr(candidate, field))
                elif field.required:
                    return None
            return dr
        return _simpleTypeConversionFn

    def _makeHardCodedTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any]):
        """This converts from one type to another when we have a hard-coded method for doing so."""
        if outputSchema == PDFFile and inputSchema == File:
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
                dr = DataRecord(outputSchema)
                dr.filename = pdf_filename
                dr.contents = pdf_bytes
                dr.text_contents = text_content
                return dr
            return _fileToPDF
        elif outputSchema == TextFile and inputSchema == File:
            def _fileToText(candidate: DataRecord):
                if not candidate.schema == inputSchema:
                    return None
                text_content = str(candidate.contents, 'utf-8')
                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                dr.contents = text_content
                return dr
            return _fileToText
        elif outputSchema == EquationImage and inputSchema == ImageFile:
            print("handling image to equation through skema")
            def _imageToEquation(candidate: DataRecord):
                if not candidate.element == inputSchema:
                    return None

                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                dr.contents = candidate.contents
                dr.equation_text = equations_to_latex(candidate.contents)
                print("Running equations_to_latex_base64: ", dr.equation_text)
                return dr
            return _imageToEquation

        elif outputSchema == ImageFile and inputSchema == File:
            def _fileToImage(candidate: DataRecord):
                if not candidate.schema == inputSchema:
                    return None
                # b64 decode of candidate.contents
                #print(candidate.contents)
                image_bytes = base64.b64encode(candidate.contents).decode('utf-8')
                #image_bytes = candidate.contents #base64.b64decode(candidate.contents)#.decode("utf-8")
                dr = DataRecord(outputSchema)
                dr.filename = candidate.filename
                if 'OPENAI_API_KEY' not in os.environ:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                # get openai key from environment
                openai_key = os.environ['OPENAI_API_KEY']
                # TODO: consider multiple image models
                dr.contents = candidate.contents
                dr.text_description = do_image_analysis(openai_key, image_bytes)
                return dr
            return _fileToImage

        else:
            raise Exception(f"Cannot hard-code conversion from {inputSchema} to {outputSchema}")

    def _makeLLMTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, conversionDesc: str):
            llmservice = config.get("llmservice", "openai")
            def fn(candidate: DataRecord):
                # iterate through all empty fields in the outputSchema and ask questions to fill them
                # for field in inputSchema.__dict__:
                dr = DataRecord(outputSchema)
                text_content = candidate.asTextJSON()
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()

                for field_name in outputSchema.fieldNames():
                    f = getattr(outputSchema, field_name)
                    try:
                        # TODO: allow for mult. fcns
                        if prompt_strategy == PromptStrategy.DSPY_COT:
                            answer = run_cot_qa(text_content,
                                                f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.",
                                                model.value,
                                                llmService=llmservice, verbose=self._verbose, promptSignature=gen_qa_signature_class(doc_schema, doc_type))
                        # TODO
                        elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                            raise Exception("not implemented yet")
                        # TODO
                        elif prompt_strategy == PromptStrategy.FEW_SHOT:
                            raise Exception("not implemented yet")

                        setattr(dr, field_name, answer)
                    except Exception as e:
                        print(f"Error: {e}")
                        setattr(dr, field_name, None)
                return dr
            return fn

    def _makeFilterFn(self, inputSchema: Schema, filter: Filter, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy):
            # parse inputs
            llmservice = config.get("llmservice", "openai")
            doc_schema = str(inputSchema)
            doc_type = inputSchema.className()

            # By default, a filter requires an LLM invocation to run
            # Someday maybe we will offer the user the chance to run a hard-coded function.
            # Or maybe we will ask the LLM to synthesize traditional code here.
            def createLLMFilter(filterCondition: str):
                def llmFilter(candidate: DataRecord):
                    if not candidate.schema == inputSchema:
                        return False
                    text_content = candidate.asTextJSON()
                    # TODO: allow for mult. fcns
                    response = None
                    if prompt_strategy == PromptStrategy.DSPY_BOOL:
                        response = run_cot_bool(text_content, filterCondition, model=model.value, llmService=llmservice,
                                                verbose=self._verbose, promptSignature=gen_filter_signature_class(doc_schema, doc_type))
                    # TODO
                    elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                        raise Exception("not implemented yet")
                    # TODO
                    elif prompt_strategy == PromptStrategy.FEW_SHOT:
                        raise Exception("not implemented yet")
                    if response == "TRUE":
                        return True
                    else:
                        return False
                return llmFilter
            return createLLMFilter(str(filter))

    def synthesize(self, taskDescriptor: Any, config: Dict[str, Any]):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputSchema, inputSchema = taskDescriptor

        if functionName == "InduceFromCandidateOp" or functionName == "ParallelInduceFromCandidateOp":
            model, prompt_strategy, conversionDesc = functionParams
            typeConversionDescriptor = (outputSchema, inputSchema)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(outputSchema, inputSchema)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(outputSchema, inputSchema, config) # TODO: add option for model for image?
            else:
                return self._makeLLMTypeConversionFn(outputSchema, inputSchema, config, model, prompt_strategy, conversionDesc)
        elif functionName == "FilterCandidateOp" or functionName == "ParallelFilterCandidateOp":
            filter, model, prompt_strategy = functionParams
            return  self._makeFilterFn(inputSchema, filter, config, model, prompt_strategy)
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
