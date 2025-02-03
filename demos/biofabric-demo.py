#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
Make sure to run:
python src/cli/cli_main.py reg --path testdata/biofabric-urls/ --name biofabric-urls

"""

import argparse
import os
import time

from palimpzest.constants import Cardinality
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils import udfs

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


CaseDataCols = [
    {"name": "case_submitter_id", "type": "string", "desc": "The ID of the case"},
    {"name": "age_at_diagnosis", "type": "string", "desc": "The age of the patient at the time of diagnosis"},
    {"name": "race", "type": "string", "desc": "An arbitrary classification of a taxonomic group that is a division of a species."},
    {"name": "ethnicity", "type": "string", "desc": "Whether an individual describes themselves as Hispanic or Latino or not."},
    {"name": "gender", "type": "string", "desc": "Text designations that identify gender."},
    {"name": "vital_status", "type": "string", "desc": "The vital status of the patient"},
    {"name": "ajcc_pathologic_t", "type": "string", "desc": "The AJCC pathologic T"},
    {"name": "ajcc_pathologic_n", "type": "string", "desc": "The AJCC pathologic N"},
    {"name": "ajcc_pathologic_stage", "type": "string", "desc": "The AJCC pathologic stage"},
    {"name": "tumor_grade", "type": "string", "desc": "The tumor grade"},
    {"name": "tumor_focality", "type": "string", "desc": "The tumor focality"}, 
    {"name": "tumor_largest_dimension_diameter", "type": "string", "desc": "The tumor largest dimension diameter"}, 
    {"name": "primary_diagnosis", "type": "string", "desc": "The primary diagnosis"},
    {"name": "morphology", "type": "string", "desc": "The morphology"},
    {"name": "tissue_or_organ_of_origin", "type": "string", "desc": "The tissue or organ of origin"},
    {"name": "tumor_code", "type": "string", "desc": "The tumor code"},
    {"name": "study", "type": "string", "desc": "The study"},
]

WebPageCols = [
    {"name": "text", "type": "string", "desc": "The text contents of the web page"},
    {"name": "html", "type": "string", "desc": "The html contents of the web page"},
    {"name": "timestamp", "type": "string", "desc": "The timestamp of the download"},
]

URLCols = [
    {"name": "url", "type": "string", "desc": "The URL of the web page"},
]

FileCols = [
    {"name": "filename", "type": "string", "desc": "The name of the file"},
    {"name": "contents", "type": "bytes", "desc": "The contents of the file"}
]

XLSCols = FileCols + [
    {"name": "number_sheets", "type": "number", "desc": "The number of sheets in the Excel file"},
    {"name": "sheet_names", "type": "list", "desc": "The names of the sheets in the Excel file"},
] 

TableCols = [
    {"name": "rows", "type": "list", "desc": "The rows of the table"},
    {"name": "header", "type": "list", "desc": "The header of the table"},
    {"name": "name", "type": "string", "desc": "The name of the table"},
    {"name": "filename", "type": "string", "desc": "The name of the file the table was extracted from"}
]


def print_table(output):
    for table in output:
        header = table.header
        subset_rows = table.rows[:3]

        print("Table name:", table.name)
        print(" | ".join(header)[:100], "...")
        for row in subset_rows:
            print(" | ".join(row)[:100], "...")
        print()


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached results")
    parser.add_argument("--verbose", action="store_true", help="Do not use cached results", default=True)
    parser.add_argument("--from_xls", action="store_true", help="Start from pre-downloaded excel files", default=False)
    parser.add_argument("--experiment", type=str, help="The experiment to run", default="matching")
    parser.add_argument("--policy", type=str, help="The policy to use", default="cost")
    parser.add_argument("--executor", type=str, help="The plan executor to use. The avaliable executors are: sequential, pipelined_parallel, pipelined_single_thread", default="pipelined_parallel")

    args = parser.parse_args()
    no_cache = args.no_cache
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy
    experiment = args.experiment
    executor = args.executor

    if no_cache:
        DataDirectory().clear_cache(keep_registry=True)

    if policy == "cost":
        policy = MinCost()
    elif policy == "quality":
        policy = MaxQuality()

    if experiment == "collection":
        papers_html = Dataset("biofabric-html")
        table_urls = papers_html.sem_add_columns(URLCols, cardinality=Cardinality.ONE_TO_MANY)
        output = table_urls

        # urlFile = Dataset("biofabric-urls", schema=TextFile)
        # table_urls = table_urls.convert(URL, desc="The URLs of the tables")
        # tables = table_urls.convert(File, udf=udfs.url_to_file)
        # xls = tables.convert(XLSFile, udf = udfs.file_to_xls)
        # patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        # output = patient_tables

    elif experiment == "filtering":
        xls = Dataset("biofabric-tiny")
        xls = xls.add_columns(udf=udfs.file_to_xls, types=XLSCols)
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, types=TableCols, cardinality=Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.sem_filter("The rows of the table contain the patient age")
        # patient_tables = patient_tables.sem_filter("The table explains the meaning of attributes")
        # patient_tables = patient_tables.sem_filter("The table contains patient biometric data")
        # patient_tables = patient_tables.sem_filter("The table contains proteomic data")
        # patient_tables = patient_tables.sem_filter("The table records if the patient is excluded from the study")
        output = patient_tables

    elif experiment == "matching":
        xls = Dataset("biofabric-matching")
        xls = xls.add_columns(udf=udfs.file_to_xls, types=XLSCols)
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, types=TableCols, cardinality=Cardinality.ONE_TO_MANY)
        case_data = patient_tables.sem_add_columns(CaseDataCols, cardinality=Cardinality.ONE_TO_MANY)
        output = case_data

    elif experiment == "endtoend":
        xls = Dataset("biofabric-tiny")
        xls = xls.add_columns(udf=udfs.file_to_xls, types=XLSCols)
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, types=TableCols, cardinality=Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.sem_filter("The rows of the table contain the patient age")
        case_data = patient_tables.sem_add_columns(CaseDataCols, cardinality=Cardinality.ONE_TO_MANY)
        output = case_data

    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy="no_sentinel",
        execution_strategy=executor,
    )
    data_record_collection = output.run(config)

    print(data_record_collection.to_df())
    print(data_record_collection.executed_plans)
    # print(data_record_collection.execution_stats)

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
