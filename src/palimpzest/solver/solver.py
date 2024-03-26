from palimpzest.constants import Model, PromptStrategy
from palimpzest.elements import DataRecord, EquationImage, File, Filter, ImageFile, PDFFile, Schema, TextFile
from palimpzest.tools.dspysearch import run_cot_bool, run_cot_qa, gen_filter_signature_class, gen_qa_signature_class
from palimpzest.tools.generic_image_converter import describe_image
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.profiler import Profiler
from palimpzest.tools.skema_tools import equations_to_latex_base64, equations_to_latex

from copy import deepcopy
from papermage import Document
from typing import Any, Dict, Tuple, Union

import json
import base64
import modal
import os

# DEFINITIONS
# TaskDescriptor = Tuple[str, Union[tuple, None], Schema, Schema]


class Solver:
    """This solves for needed operator implementations"""
    def __init__(self, verbose: bool=False):
        self._hardcodedFns = {}
        self._simpleTypeConversions = set()
        self._hardcodedFns = set()
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((PDFFile, File))
        self._hardcodedFns.add((TextFile, File))
        # self._hardcodedFns.add((ImageFile, File))
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

    def _makeHardCodedTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], op_id: str, shouldProfile=False):
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
            # TODO: use op_id and shouldProfile to measure time spent waiting on API response
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
        else:
            raise Exception(f"Cannot hard-code conversion from {inputSchema} to {outputSchema}")

    def _makeLLMTypeConversionFn(self, outputSchema: Schema, inputSchema: Schema, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, op_id: str, conversionDesc: str, shouldProfile=False):
            def fn(candidate: DataRecord):
                # iterate through all empty fields in the outputSchema and ask questions to fill them
                # for field in inputSchema.__dict__:
                text_content = candidate.asTextJSON()
                doc_schema = str(outputSchema)
                doc_type = outputSchema.className()
                stats = {}

                def runBondedQuery():
                    dr = DataRecord(outputSchema)
                    field_stats = None

                    fieldDescriptions = {}
                    for field_name in outputSchema.fieldNames():
                        f = getattr(outputSchema, field_name)
                        fieldDescriptions[field_name] = f.desc

                    multilineInputFieldDescription = ""
                    for field_name in inputSchema.fieldNames():
                        f = getattr(inputSchema, field_name)
                        multilineInputFieldDescription += f"INPUT FIELD {field_name}: {f.desc}\n"

                    multilineOutputFieldDescription = ""
                    for field_name in outputSchema.fieldNames():
                        f = getattr(outputSchema, field_name)
                        multilineOutputFieldDescription += f"OUTPUT FIELD {field_name}: {f.desc}\n"

                    optionalInputDesc = "" if inputSchema.__doc__ is None else f"Here is a description of the input object: {inputSchema.__doc__}."
                    optionalOutputDesc = "" if outputSchema.__doc__ is None else f"Here is a description of the output object: {outputSchema.__doc__}."
                    promptQuestion = f"""I would like you to create a output JSON object that describes an object of type {doc_type}. 
                    You will use the information in an input JSON object that I will provide. The input object has type {inputSchema.className()}.
                    All of the fields in the output object can be derived using information from the input object.
                    {optionalInputDesc}
                    {optionalOutputDesc}
                    Here is every input field name and a description: 
                    {multilineInputFieldDescription}
                    Here is every output field name and a description:
                    {multilineOutputFieldDescription}.
                    Be sure to emit a JSON object only.
                    """ + "" if conversionDesc is None else f" Keep in mind that this process is described by this text: {conversionDesc}."                

                    answer = None
                    if prompt_strategy == PromptStrategy.DSPY_COT:
                        answer, field_stats = run_cot_qa(text_content, promptQuestion,
                                                                model_name=model.value, 
                                                                verbose=self._verbose, 
                                                                promptSignature=gen_qa_signature_class(doc_schema, doc_type),
                                                                shouldProfile=shouldProfile)
                    if not answer.strip().startswith('{'):
                        # Find the start index of the actual JSON string
                        # assuming the prefix is followed by the JSON object/array
                        start_index = answer.find('{') if '{' in answer else answer.find('[')
                        if start_index != -1:
                            # Remove the prefix and any leading characters before the JSON starts
                            answer = answer[start_index:]
                    if not answer.strip().endswith('}'):
                        # Find the end index of the actual JSON string
                        # assuming the suffix is preceded by the JSON object/array
                        end_index = answer.rfind('}') if '}' in answer else answer.rfind(']')
                        if end_index != -1:
                            # Remove the suffix and any trailing characters after the JSON ends
                            answer = answer[:end_index + 1]

                    # Handle weird escaped values. I am not sure why the model
                    # is returning these, but the JSON parser can't take them
                    answer = answer.replace("\_", "_")
                    jsonObj = json.loads(answer)
                    for field_name in outputSchema.fieldNames():
                        # parse the json object and set the fields of the record
                        setattr(dr, field_name, jsonObj[field_name])
                        stats[f"{field_name}"] = field_stats

                    # if profiling, set record's stats for the given op_id
                    if shouldProfile:
                        dr._stats[op_id] = {"fields": stats}

                    return dr

                def runConventionalQuery():
                    dr = DataRecord(outputSchema)
                    for field_name in outputSchema.fieldNames():
                        f = getattr(outputSchema, field_name)
                        try:
                            # TODO: allow for mult. fcns
                            field_stats = None
                            if prompt_strategy == PromptStrategy.DSPY_COT:                            
                                #print("ABOUT TO RUN", text_content, f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.")
                                #print("About to run model", model.value)
                                answer, field_stats = run_cot_qa(text_content,
                                                                f"What is the {field_name} of the {doc_type}? ({f.desc})" + "" if conversionDesc is None else f" Keep in mind that this output is described by this text: {conversionDesc}.",
                                                                model_name=model.value,
                                                                verbose=self._verbose,
                                                                promptSignature=gen_qa_signature_class(doc_schema, doc_type),
                                                                shouldProfile=shouldProfile)

                            elif prompt_strategy == PromptStrategy.IMAGE_TO_TEXT:                               
                                    if not candidate.schema == inputSchema:
                                        return None

                                    # b64 decode of candidate.contents
                                    image_b64 = base64.b64encode(candidate.contents).decode('utf-8')
                                    dr = DataRecord(outputSchema)
                                    dr.filename = candidate.filename
                                    dr.contents = candidate.contents
                                    answer, field_stats = describe_image(model_name=model.value, image_b64=image_b64, shouldProfile=shouldProfile)

                                    # if profiling, set record's stats for the given op_id
                                    if shouldProfile:
                                        field_stats[op_id] = {"fields": {"text_description": stats}}

                            # TODO
                            elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                                raise Exception("not implemented yet")
                            # TODO
                            elif prompt_strategy == PromptStrategy.FEW_SHOT:
                                raise Exception("not implemented yet")

                            setattr(dr, field_name, answer)
                            stats[f"{field_name}"] = field_stats
                        except Exception as e:
                            print(f"Traditional field processing error: {e}")
                            setattr(dr, field_name, None)
                
                    # if profiling, set record's stats for the given op_id
                    if shouldProfile:
                        dr._stats[op_id] = {"fields": stats}

                    return dr

                try:
                    return runBondedQuery()
                except Exception as e:
                    try:
                        return runConventionalQuery()
                    except Exception as e:
                        print(f"Error: {e}")
                        dr = DataRecord(outputSchema)
                        for field_name in outputSchema.fieldNames():
                            setattr(dr, field_name, None)
                        return dr
            return fn

    def _makeFilterFn(self, inputSchema: Schema, filter: Filter, config: Dict[str, Any], model: Model, prompt_strategy: PromptStrategy, op_id: str, shouldProfile=False):
            # parse inputs
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
                    response, stats = None, {}
                    if prompt_strategy == PromptStrategy.DSPY_BOOL:
                        response, stats = run_cot_bool(text_content, 
                                                       filterCondition, 
                                                       model_name=model.value,
                                                       verbose=self._verbose, 
                                                       promptSignature=gen_filter_signature_class(doc_schema, doc_type),
                                                       shouldProfile=shouldProfile)
                    # TODO
                    elif prompt_strategy == PromptStrategy.ZERO_SHOT:
                        raise Exception("not implemented yet")
                    # TODO
                    elif prompt_strategy == PromptStrategy.FEW_SHOT:
                        raise Exception("not implemented yet")

                    # if profiling, set record's stats for the given op_id and clear any state from the previous operation
                    if shouldProfile:
                        candidate = deepcopy(candidate)
                        candidate._state = {}
                        candidate._stats[op_id] = stats

                    # set _passed_filter attribute and return record
                    setattr(candidate, "_passed_filter", response.lower() == "true")

                    return candidate

                return llmFilter
            return createLLMFilter(str(filter))

    def synthesize(self, taskDescriptor: Any, config: Dict[str, Any], shouldProfile=False):
        """Return a function that maps from inputType to outputType."""
        functionName, functionParams, outputSchema, inputSchema = taskDescriptor

        if functionName == "InduceFromCandidateOp" or functionName == "ParallelInduceFromCandidateOp":
            model, prompt_strategy, op_id, conversionDesc = functionParams
            typeConversionDescriptor = (outputSchema, inputSchema)
            if typeConversionDescriptor in self._simpleTypeConversions:
                return self._makeSimpleTypeConversionFn(outputSchema, inputSchema)
            elif typeConversionDescriptor in self._hardcodedFns:
                return self._makeHardCodedTypeConversionFn(outputSchema, inputSchema, config, op_id, shouldProfile=shouldProfile)
            else:
                return self._makeLLMTypeConversionFn(outputSchema, inputSchema, config, model, prompt_strategy, op_id, conversionDesc, shouldProfile=shouldProfile)
        elif functionName == "FilterCandidateOp" or functionName == "ParallelFilterCandidateOp":
            filter, model, prompt_strategy, op_id = functionParams
            return self._makeFilterFn(inputSchema, filter, config, model, prompt_strategy, op_id, shouldProfile=shouldProfile)
        else:
            raise Exception("Cannot synthesize function for task descriptor: " + str(taskDescriptor))
