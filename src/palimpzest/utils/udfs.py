"""
This file collects a sample of useful UDFs to convert schemata.
"""

import io
import json
from datetime import datetime

import modal
import pandas as pd
import requests
from papermage import Document

from palimpzest.constants import MAX_ROWS
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.tools.pdfparser import get_text_from_pdf


def url_to_file(candidate: dict):
    """Function used to convert a DataRecord instance of URL to a File DataRecord."""
    url = candidate["url"]
    filename = url.split("/")[-1]
    timestamp = datetime.now().isoformat()
    try:
        contents = requests.get(url).content
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        contents = b""

    return {"filename": filename, "timestamp": timestamp, "contents": contents}


def file_to_pdf(candidate: dict):
    pdfprocessor = DataDirectory().current_config.get("pdfprocessor")
    if pdfprocessor == "modal":
        print("handling PDF processing remotely")
        remote_func = modal.Function.lookup("palimpzest.tools", "processPapermagePdf")
    else:
        remote_func = None

    pdf_bytes = candidate["contents"]
    # generate text_content from PDF
    if remote_func is not None:
        doc_json_str = remote_func.remote([pdf_bytes])
        docdict = json.loads(doc_json_str[0])
        doc = Document.from_json(docdict)
        text_content = ""
        for p in doc.pages:
            text_content += p.text
    else:
        text_content = get_text_from_pdf(candidate["filename"], candidate["contents"])

    return {"text_contents": text_content[:10000]}  # TODO Very hacky


def file_to_xls(candidate: dict):
    """Function used to convert a DataRecord instance of File to a XLSFile DataRecord."""
    xls = pd.ExcelFile(io.BytesIO(candidate["contents"]), engine="openpyxl")
    return {"number_sheets": len(xls.sheet_names), "sheet_names": xls.sheet_names}


def xls_to_tables(candidate: dict):
    """Function used to convert a DataRecord instance of XLSFile to a Table DataRecord."""
    xls_bytes = candidate["contents"]
    sheet_names = candidate["sheet_names"]

    records = []
    for sheet_name in sheet_names:
        dataframe = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=sheet_name, engine="openpyxl")

        # TODO extend number of rows with dynamic sizing of context length
        # construct data record
        record = {}
        rows = []
        for row in dataframe.values[:100]:
            row_record = [str(x) for x in row]
            rows += [row_record]
        record["rows"] = rows[:MAX_ROWS]
        record["filename"] = candidate["filename"]
        record["header"] = dataframe.columns.values.tolist()
        record["name"] = candidate["filename"].split("/")[-1] + "_" + sheet_name
        records.append(record)

    return records
