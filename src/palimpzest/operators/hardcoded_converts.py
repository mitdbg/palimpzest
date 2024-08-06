from __future__ import annotations
from datetime import datetime

import palimpzest as pz
from palimpzest.constants import *
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.elements import DataRecord
from palimpzest.operators import logical
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
import time

import requests


class HardcodedConvert(ConvertOp):

    implemented_op = logical.ConvertScan

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.inputSchema == self.__class__.inputSchema
            and self.outputSchema == self.__class__.outputSchema
        ), f"This convert has to be instantiated to convert a {self.__class__.inputSchema} to a {self.__class__.outputSchema}! But it was instantiated to convert a {self.inputSchema} to a {self.outputSchema}!"

    @classmethod
    def materializes(cls, logical_operator: logical.LogicalOperator):
        return (
            cls.inputSchema == logical_operator.inputSchema
            and cls.outputSchema == logical_operator.outputSchema
        )

    def __eq__(self, other: HardcodedConvert):
        return (
            isinstance(other, self.__class__)
            and self.cardinality == other.cardinality
            and self.outputSchema == other.outputSchema
            and self.inputSchema == other.inputSchema
            and self.max_workers == other.max_workers
        )

    # def __str__(self):
        # return f"{self.__class__.__name__}({str(self.outputSchema):10s})"

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
    final = True

    def __call__(self, candidate: DataRecord):
        start_time = time.time()

        text_content = candidate.contents
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = text_content

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return [dr], [record_op_stats]


class ConvertImageToEquation(HardcodedConvert):

    inputSchema = schemas.ImageFile
    outputSchema = schemas.EquationImage
    final = True

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

    def __call__(self, candidate: DataRecord):
        start_time = time.time()

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = candidate.contents

        api_start_time = time.time()
        dr.equation_text = equations_to_latex(candidate.contents)
        api_call_duration_secs = time.time() - api_start_time

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            fn_call_duration_secs=api_call_duration_secs,
        )

        return [dr], [record_op_stats]


class ConvertURLToFile(HardcodedConvert):
    """
    NOTE: I am happy leaving this as-is for now, but in the long(er) term I think
    we should look back at our demos and either:

    (A) extract the pieces that generalize across workloads, or
    (B) move demo-specific code into user-supported classes

    For example, similar to how we have user-defined DataSources, we should soon
    support a user-defined hard-coded convert operation. This class could then be
    implemented as a hard-coded convert (or be better generalized).
    """

    inputSchema = schemas.URL
    outputSchema = schemas.File
    final = True

    def __call__(self, candidate: DataRecord):
        start_time = time.time()

        # Assign a filename that is parsed from the URL
        # NOTE: will this generalize? if not, it should be moved into a user-specific class
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.url.split("/")[-1]
        dr.url = candidate.url
        dr.timestamp = datetime.now().isoformat()
        try:
            contents = requests.get(candidate.url).content
        except Exception as e:
            print(f"Error fetching URL {candidate.url}: {e}")
            contents = b''
        dr.contents = contents

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return [dr], [record_op_stats]


class ConvertFileToXLS(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.XLSFile
    final = True

    def __call__(self, candidate: DataRecord):
        start_time = time.time()

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = candidate.filename
        dr.contents = candidate.contents

        api_start_time = time.time()
        xls = pd.ExcelFile(BytesIO(candidate.contents), engine="openpyxl")
        api_call_duration_secs = time.time() - api_start_time

        dr.number_sheets = len(xls.sheet_names)
        dr.sheet_names = xls.sheet_names

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            fn_call_duration_secs=api_call_duration_secs,
        )

        return [dr], [record_op_stats]


class ConvertXLSToTable(HardcodedConvert):

    inputSchema = schemas.XLSFile
    outputSchema = schemas.Table
    final = True

    def __call__(self, candidate: DataRecord):
        start_time = time.time()
        cardinality = self.cardinality
        xls_bytes = candidate.contents
        # dr.sheets = [xls.parse(name) for name in candidate.sheet_names]
        sheet_names = (
            [candidate.sheet_names[0]] if cardinality == "oneToOne" else candidate.sheet_names
        )

        records, record_op_stats_lst = [], []
        for sheet_name in sheet_names:
            api_start_time = time.time()
            dataframe = pd.read_excel(
                BytesIO(xls_bytes), sheet_name=sheet_name, engine="openpyxl"
            )
            api_call_duration_secs = time.time() - api_start_time

            # TODO extend number of rows with dynamic sizing of context length
            # construct data record
            dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
            rows = []
            for row in dataframe.values[:100]:
                row_record = [str(x) for x in row]
                rows += [row_record]
            dr.rows = rows[:MAX_ROWS]
            dr.filename = candidate.filename
            dr.header = dataframe.columns.values.tolist()
            dr.name = candidate.filename.split("/")[-1] + "_" + sheet_name

            # create RecordOpStats object
            record_op_stats = RecordOpStats(
                record_uuid=dr._uuid,
                record_parent_uuid=dr._parent_uuid,
                record_state=dr._asDict(include_bytes=False),
                op_id=self.get_op_id(),
                op_name=self.op_name(),
                time_per_record=time.time() - start_time,
                cost_per_record=0.0,
                fn_call_duration_secs=api_call_duration_secs,
            )

            # update start_time, records, and record_op_stats_lst
            start_time = time.time()
            records.append(dr)
            record_op_stats_lst.append(record_op_stats)

        return records, record_op_stats_lst


class ConvertFileToPDF(HardcodedConvert):

    inputSchema = schemas.File
    outputSchema = schemas.PDFFile
    final = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdfprocessor = pz.DataDirectory().current_config.get("pdfprocessor"),

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

    def __call__(self, candidate: DataRecord):
        start_time = time.time()
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
        api_start_time = time.time()
        if remoteFunc is not None:
            docJsonStr = remoteFunc.remote([pdf_bytes])
            docdict = json.loads(docJsonStr[0])
            doc = Document.from_json(docdict)
            text_content = ""
            for p in doc.pages:
                text_content += p.text
        else:
            text_content = get_text_from_pdf(candidate.filename, candidate.contents)
        api_call_duration_secs = time.time() - api_start_time

        # construct data record
        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        dr.filename = pdf_filename
        dr.contents = pdf_bytes
        dr.text_contents = text_content[:10000]  # TODO Very hacky

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            fn_call_duration_secs=api_call_duration_secs,
        )

        return [dr], [record_op_stats]


class SimpleTypeConvert(HardcodedConvert):
    """This is a very simple function that converts a DataRecord from one Schema to another, when we know they have identical fields."""

    def __init__(self, *args, **kwargs):
        ConvertOp.__init__(*args, **kwargs)
        # TODO: does schema equality only check fields?
        assert (
            self.inputSchema == self.outputSchema
        ), "This convert has to be instantiated to convert an input to the same output Schema!"

    def __call__(self, candidate: DataRecord):
        start_time = time.time()

        dr = DataRecord(self.outputSchema, parent_uuid=candidate._uuid)
        for field in self.outputSchema.fieldNames():  # type: ignore
            if hasattr(candidate, field):
                setattr(dr, field, getattr(candidate, field))
            elif field.required:
                return None

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_uuid=dr._uuid,
            record_parent_uuid=dr._parent_uuid,
            record_state=dr._asDict(include_bytes=False),
            op_id=self.get_op_id(),
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return [dr], [record_op_stats]
