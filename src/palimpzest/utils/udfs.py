"""
This file collects a sample of useful UDFs to convert schemata.
"""

import io
from datetime import datetime
from selenium import webdriver

import pandas as pd
import requests

from palimpzest.constants import MAX_ROWS


def url_to_file(candidate: dict):
    """Function used to convert a DataRecord instance of URL to a File DataRecord."""

    url = candidate["url"]
    if isinstance(url, list) and len(url) > 0:
        url = str(url[0])
    elif not url:
        return {"filename": "", "timestamp": "", "contents": b""}
    filename = url.split("/")[-1]
    timestamp = datetime.now().isoformat()
    try:
        if not url.startswith("http"):
            url = "https://www.cell.com" + url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        contents = requests.get(url, headers=headers, allow_redirects=True).content
        # contents = browser.get(url).page_source
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        contents = b""

    return {"filename": filename, "timestamp": timestamp, "contents": contents}


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
