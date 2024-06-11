from __future__ import annotations

from palimpzest.constants import *
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord
from palimpzest.operators.convert import ConvertOp
from palimpzest.tools.pdfparser import get_text_from_pdf
from palimpzest.tools.skema_tools import equations_to_latex

import palimpzest.corelib.schemas as schemas

from io import BytesIO
from papermage import Document
from typing import Optional

import pandas as pd

import json
import modal


class HardcodedConvert(ConvertOp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.inputSchema == self.__class__.inputSchema
            and self.outputSchema == self.__class__.outputSchema
        ), f"This convert has to be instantiated to convert a {self.__class__.inputSchema} to a {self.__class__.outputSchema}! But it was instantiated to convert a {self.inputSchema} to a {self.outputSchema}!"

    def __eq__(self, other: HardcodedConvert):
        return (
            isinstance(other, self.__class__)
            and self.cardinality == other.cardinality
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.outputSchema):10s})"

    def copy(self):
        return self.__class__(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            cardinality=self.cardinality,
            desc=self.desc,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        cardinality = (
            source_op_cost_estimates.cardinality
            if self.cardinality == "oneToOne"
            else NAIVE_EST_ONE_TO_MANY_SELECTIVITY * source_op_cost_estimates.cardinality
        )
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=0.0,
            cost_per_record=0.0,
            quality=1.0,
        )

    def __call__(self, candidate: DataRecord):
        raise NotImplementedError("This is an abstract class. Use a subclass instead.")
    
    def is_hardcoded(self) -> bool:
        return True


class ConvertFileToText(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.TextFile

    # TODO
    def __call__(self, candidate: DataRecord):

        if not candidate.schema == self.inputSchema:
            return None
        text_content = str(candidate.contents, "utf-8")
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = text_content
        # if profiling, set record's stats for the given op_id to be an empty Stats object
        # if shouldProfile:
        # candidate._stats[td.op_id] = ConvertNonLLMStats()

        return [dr], None


class ConvertImageToEquation(HardcodedConvert):

    inputSchema = schemas.ImageFile
    outputSchema = schemas.EquationImage

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        cardinality = (
            source_op_cost_estimates.cardinality
            if self.cardinality == "oneToOne"
            else NAIVE_EST_ONE_TO_MANY_SELECTIVITY * source_op_cost_estimates.cardinality
        )
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=NAIVE_IMAGE_TO_EQUATION_LATEX_TIME_PER_RECORD,
            cost_per_record=0.0,
            quality=1.0,
        )

    # TODO
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
        # dr._stats[td.op_id] = ConvertNonLLMStats(api_stats=api_stats)
        return [dr], None


class ConvertDownloadToFile(HardcodedConvert):

    inputSchema = schemas.Download
    outputSchema = schemas.File

    # TODO
    def __call__(self, candidate: DataRecord):
        if not candidate.schema == self.inputSchema:
            return None
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        # Assign a filename that is parsed from the URL
        dr.filename = candidate.url.split("/")[-1]
        dr.contents = candidate.content
        # if shouldProfile:
        # candidate._stats[td.op_id] = ConvertNonLLMStats()
        return [dr], None


class ConvertFileToXLS(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.XLSFile

    # TODO
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
        # candidate._stats[td.op_id] = ConvertNonLLMStats(api_stats=api_stats)


class ConvertXLSToTable(HardcodedConvert):

    inputSchema = schemas.XLSFile
    outputSchema = schemas.Table

    # TODO
    def __call__(self, candidate: DataRecord):
        cardinality = self.cardinality

        xls_bytes = candidate.contents
        # dr.sheets = [xls.parse(name) for name in candidate.sheet_names]
        sheet_names = (
            [candidate.sheet_names[0]] if cardinality == "oneToOne" else candidate.sheet_names
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
            # dr._stats[td.op_id] = ConvertNonLLMStats(api_stats=api_stats)
            records.append(dr)
        return records, None


class ConvertFileToPDF(HardcodedConvert):
    inputSchema = schemas.File
    outputSchema = schemas.PDFFile

    def __init__(self, pdfprocessor: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdfprocessor = pdfprocessor

    def __eq__(self, other: ConvertFileToPDF):
        return (
            isinstance(other, self.__class__)
            and self.cardinality == other.cardinality
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
            and self.pdfprocessor == other.pdfprocessor
        )

    def copy(self):
        return self.__class__(
            outputSchema=self.outputSchema,
            inputSchema=self.inputSchema,
            cardinality=self.cardinality,
            desc=self.desc,
            pdfprocessor=self.pdfprocessor,
            targetCacheId=self.targetCacheId,
            shouldProfile=self.shouldProfile,
        )

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        cardinality = (
            source_op_cost_estimates.cardinality
            if self.cardinality == "oneToOne"
            else NAIVE_EST_ONE_TO_MANY_SELECTIVITY * source_op_cost_estimates.cardinality
        )
        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=NAIVE_PDF_PROCESSOR_TIME_PER_RECORD,
            cost_per_record=0.0,
            quality=1.0,
        )

    # TODO
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
        # dr._stats[td.op_id] = ConvertNonLLMStats(api_stats=api_stats)
        return [dr], None


# NOTE 1: My motivation for making this inherit from HardcodedConvert is that I have at least one place
#         in the CostEstimator where it would be nice to simply check if an operator is an instance of HardcodedConvert
# NOTE 2: I realize that this does not match the convention of having an inputSchema
#         and outputSchema which can be checked against self.__class__.inputSchema
#         and self.__class__.outputSchema. But, since it is a hard-coded convert operation,
#         I feel like having this inherit from HardcodedConvert and overriding the __init__
#         is a middle-ground
class SimpleTypeConvert(HardcodedConvert):
    """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""

    def __init__(self, *args, **kwargs):
        ConvertOp.__init__(*args, **kwargs)
        # TODO: does schema equality only check fields?
        assert (
            self.inputSchema == self.outputSchema
        ), "This convert has to be instantiated to convert an input to the same output Schema!"

    # TODO
    def __call__(self, candidate: DataRecord):
        if not candidate.schema == self.inputSchema:
            return None

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        for field in self.outputSchema.fieldNames():  # type: ignore

            # TODO: I think we can drop the if/elif since we check candidate.schema == self.inputSchema above?
            if hasattr(candidate, field):
                setattr(dr, field, getattr(candidate, field))
            elif field.required:
                return None

        # TODO profiling should be done somewhere else
        # if profiling, set record's stats for the given op_id to be an empty Stats object
        # if self.shouldProfile:
        # candidate._stats[td.op_id] = ConvertNonLLMStats()

        return [dr], None