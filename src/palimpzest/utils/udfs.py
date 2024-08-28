"""
This scripts collects a sample of useful UDFs to convert schemata.
"""
import io
import palimpzest as pz
from palimpzest.elements.records import DataRecord
import pandas as pd

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

from papermage import Document
from typing import Optional

import pandas as pd

import json
import modal
import time

import requests
from requests_html import HTMLSession


def file_to_pdf(candidate):
    pdfprocessor = pz.DataDirectory().current_config.get("pdfprocessor")
    if pdfprocessor == "modal":
        print("handling PDF processing remotely")
        remoteFunc = modal.Function.lookup(
            "palimpzest.tools", "processPapermagePdf"
        )
    else:
        remoteFunc = None

    pdf_bytes = candidate.contents
    # generate text_content from PDF
    if remoteFunc is not None:
        docJsonStr = remoteFunc.remote([pdf_bytes])
        docdict = json.loads(docJsonStr[0])
        doc = Document.from_json(docdict)
        text_content = ""
        for p in doc.pages:
            text_content += p.text
    else:
        text_content = get_text_from_pdf(candidate.filename, candidate.contents)

    # construct data record
    candidate.text_contents = text_content[:10000]  # TODO Very hacky

    return [candidate]

def file_to_xls(candidate):
    """ Function used to convert a DataRecord instance of File to a XLSFile DataRecord. """
    xls = pd.ExcelFile(io.BytesIO(candidate.contents), engine="openpyxl")
    candidate.number_sheets = len(xls.sheet_names)
    candidate.sheet_names = xls.sheet_names
    return [candidate]
    
def xls_to_tables(candidate):
    """ Function used to convert a DataRecord instance of XLSFile to a Table DataRecord. """
    xls_bytes = candidate.contents
    sheet_names = candidate.sheet_names

    records = []
    for sheet_name in sheet_names:
        dataframe = pd.read_excel(
            io.BytesIO(xls_bytes), sheet_name=sheet_name, engine="openpyxl"
        )

        # TODO extend number of rows with dynamic sizing of context length
        # construct data record
        dr = pz.DataRecord(pz.Table, parent_id=candidate._id)
        rows = []
        for row in dataframe.values[:100]:
            row_record = [str(x) for x in row]
            rows += [row_record]
        dr.rows = rows[pz.MAX_ROWS]
        dr.filename = candidate.filename
        dr.header = dataframe.columns.values.tolist()
        dr.name = candidate.filename.split("/")[-1] + "_" + sheet_name
        records.append(dr)

    return records

def url_to_file(candidate):
    """ Function used to convert a DataRecord instance of URL to a File DataRecord. """
    candidate.filename = candidate.url.split("/")[-1]
    candidate.timestamp = datetime.now().isoformat()
    try:
        contents = requests.get(candidate.url).content
    except Exception as e:
        print(f"Error fetching URL {candidate.url}: {e}")
        contents = b''
    candidate.contents = contents
    return [candidate]
