from __future__ import annotations
from io import BytesIO

from src.palimpzest.operators.convert import ConvertOp
import palimpzest.corelib.schemas as schemas

# from palimpzest.constants import *
from palimpzest.elements import DataRecord

from typing import Any, Dict, Optional, Tuple

import json
import modal
import pandas as pd
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.skema_tools import equations_to_latex
from papermage import Document

# TODO:
# 1. ensure that __init__(self, *args, **kwargs) is also in ConvertOp to allow e.g. self.pdfprocessor to be set
# 2. rewrite .schema in records.py to return dynamic set of fields
# 3.


class HardcodedConvert(ConvertOp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.inputSchema == self.__class__.inputSchema
            and self.outputSchema == self.__class__.outputSchema
        ), f"This convert has to be instantiated to convert a {self.__class__.inputSchema} to a {self.__class__.outputSchema}!"

    def __call__(self, candidate: DataRecord):
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")


class ConvertFileToText(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.TextFile

    def __call__(self, candidate: DataRecord):

        if not candidate.schema == self.inputSchema:
            return None
        text_content = str(candidate.contents, "utf-8")
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = text_content
        # if profiling, set record's stats for the given op_id to be an empty Stats object
        # if shouldProfile:
        # candidate._stats[td.op_id] = InduceNonLLMStats()

        return [dr], None


class ConvertImageToEquation(HardcodedConvert):

    inputSchema = schemas.ImageFile
    outputSchema = schemas.EquationImage

    def __call__(self, candidate: DataRecord):
        print("handling image to equation through skema")
        if not candidate.schema == self.inputSchema:
            return None

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = candidate.contents
        dr.equation_text, api_stats = equations_to_latex(candidate.contents)
        print("Running equations_to_latex_base64: ", dr.equation_text)
        # if profiling, set record's stats for the given op_id
        # if shouldProfile:
        # dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
        return [dr], None


class ConvertDownloadToFile(HardcodedConvert):

    inputSchema = schemas.Download
    outputSchema = schemas.File

    def __call__(self, candidate: DataRecord):
        if not candidate.schema == self.inputSchema:
            return None
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        # Assign a filename that is parsed from the URL
        dr.filename = candidate.url.split("/")[-1]
        dr.contents = candidate.content
        # if shouldProfile:
        # candidate._stats[td.op_id] = InduceNonLLMStats()
        return [dr], None


class ConvertFileToXLS(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.XLSFile

    def __call__(self, candidate: DataRecord):
        if not candidate.schema == self.inputSchema:
            return None
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = candidate.contents
        # start_time = time.time()
        xls = pd.ExcelFile(BytesIO(candidate.contents), engine="openpyxl")
        # api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)
        dr.number_sheets = len(xls.sheet_names)
        dr.sheet_names = xls.sheet_names
        # if shouldProfile:
        # candidate._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)


class ConvertXLSToTable(HardcodedConvert):

    inputSchema = schemas.XLSFile
    outputSchema = schemas.Table

    def __call__(self, candidate: DataRecord):
        cardinality = self.cardinality

        xls_bytes = candidate.contents
        # dr.sheets = [xls.parse(name) for name in candidate.sheet_names]
        sheet_names = (
            [candidate.sheet_names[0]] if cardinality is None else candidate.sheet_names
        )

        records = []
        for sheet_name in sheet_names:
            # start_time = time.time()
            dataframe = pd.read_excel(
                BytesIO(xls_bytes), sheet_name=sheet_name, engine="openpyxl"
            )
            # api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)

            # TODO extend number of rows with dynamic sizing of context length
            # construct data record
            dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
            rows = []
            for row in dataframe.values[:100]:
                row_record = [str(x) for x in row]
                rows += [row_record]
            dr.rows = rows
            dr.filename = candidate.filename
            dr.header = dataframe.columns.values.tolist()
            dr.name = candidate.filename.split("/")[-1] + "_" + sheet_name

            # if shouldProfile:
            # dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
            records.append(dr)
        return records, None


class ConvertFileToPDF(HardcodedConvert):
    inputSchema = schemas.File
    outputSchema = schemas.PDFFile

    def __call__(self, candidate: DataRecord):
        if self.pdfprocessor == "modal":
            print("handling PDF processing remotely")
            remoteFunc = modal.Function.lookup(
                "palimpzest.tools", "processPapermagePdf"
            )
        else:
            remoteFunc = None

        # parse PDF variables
        pdf_bytes = candidate.contents
        pdf_filename = candidate.filename

        # generate text_content from PDF
        # start_time = time.time()
        if remoteFunc is not None:
            docJsonStr = remoteFunc.remote([pdf_bytes])
            docdict = json.loads(docJsonStr[0])
            doc = Document.from_json(docdict)
            text_content = ""
            for p in doc.pages:
                text_content += p.text
        else:
            text_content = get_text_from_pdf(candidate.filename, candidate.contents)

        # construct an ApiStats object to reflect time spent waiting
        # api_stats = ApiStats(api_call_duration_secs=time.time() - start_time)

        # construct data record
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = pdf_filename
        dr.contents = pdf_bytes
        dr.text_contents = text_content[:10000]  # TODO Very hacky
        # if profiling, set record's stats for the given op_id
        # if shouldProfile:
        # dr._stats[td.op_id] = InduceNonLLMStats(api_stats=api_stats)
        return [dr], None
