#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
Make sure to run:
python src/cli/cli_main.py reg --path testdata/biofabric-urls/ --name biofabric-urls

"""

import argparse
import os
import time

import palimpzest as pz
from palimpzest.utils import udfs

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


case_data_cols = [
    {"name": "case_submitter_id", "type": str, "desc": "The ID of the case"},
    {"name": "age_at_diagnosis", "type": int | float, "desc": "The age of the patient at the time of diagnosis"},
    {"name": "race", "type": str, "desc": "An arbitrary classification of a taxonomic group that is a division of a species."},
    {"name": "ethnicity", "type": str, "desc": "Whether an individual describes themselves as Hispanic or Latino or not."},
    {"name": "gender", "type": str, "desc": "Text designations that identify gender."},
    {"name": "vital_status", "type": str, "desc": "The vital status of the patient"},
    {"name": "ajcc_pathologic_t", "type": str, "desc": "Code of pathological T (primary tumor) to define the size or contiguous extension of the primary tumor (T), using staging criteria from the American Joint Committee on Cancer (AJCC)."},
    {"name": "ajcc_pathologic_n", "type": str, "desc": "The codes that represent the stage of cancer based on the nodes present (N stage) according to criteria based on multiple editions of the AJCC's Cancer Staging Manual."},
    {"name": "ajcc_pathologic_stage", "type": str, "desc": "The extent of a cancer, especially whether the disease has spread from the original site to other parts of the body based on AJCC staging criteria."},
    {"name": "tumor_grade", "type": int | float, "desc": "Numeric value to express the degree of abnormality of cancer cells, a measure of differentiation and aggressiveness."},
    {"name": "tumor_focality", "type": str, "desc": "The text term used to describe whether the patient's disease originated in a single location or multiple locations."},
    {"name": "tumor_largest_dimension_diameter", "type": int | float, "desc": "The tumor largest dimension diameter."},
    {"name": "primary_diagnosis", "type": str, "desc": "Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "morphology", "type": str, "desc": "The Morphological code of the tumor, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "tissue_or_organ_of_origin", "type": str, "desc": "The text term used to describe the anatomic site of origin, of the patient's malignant disease, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O)."},
    {"name": "study", "type": str, "desc": "The last name of the author of the study, from the table name"}
]

web_page_cols = [
    {"name": "text", "type": str, "desc": "The text contents of the web page"},
    {"name": "html", "type": str, "desc": "The html contents of the web page"},
    {"name": "timestamp", "type": str, "desc": "The timestamp of the download"},
]

url_cols = [
    {"name": "url", "type": str, "desc": "The URL of the web page"},
]

file_cols = [
    {"name": "filename", "type": str, "desc": "The name of the file"},
    {"name": "contents", "type": bytes, "desc": "The contents of the file"}
]

xls_cols = file_cols + [
    {"name": "number_sheets", "type": int, "desc": "The number of sheets in the Excel file"},
    {"name": "sheet_names", "type": list[str], "desc": "The names of the sheets in the Excel file"},
]

table_cols = [
    {"name": "rows", "type": list[str], "desc": "The rows of the table"},
    {"name": "header", "type": list[str], "desc": "The header of the table"},
    {"name": "name", "type": str, "desc": "The name of the table"},
    {"name": "filename", "type": str, "desc": "The name of the file the table was extracted from"}
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
    parser.add_argument("--verbose", action="store_true", help="Do not use cached results", default=True)
    parser.add_argument("--from_xls", action="store_true", help="Start from pre-downloaded excel files", default=False)
    parser.add_argument("--experiment", type=str, help="The experiment to run", default="matching")
    parser.add_argument("--policy", type=str, help="The policy to use", default="cost")
    parser.add_argument("--executor", type=str, help="The plan executor to use. The avaliable executors are: sequential, pipelined, parallel", default="parallel")

    args = parser.parse_args()
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy
    experiment = args.experiment
    executor = args.executor

    if policy == "cost":
        policy = pz.MinCost()
    elif policy == "quality":
        policy = pz.MaxQuality()

    if experiment == "collection":
        papers_html = pz.Dataset("testdata/biofabric-html")
        papers_html = papers_html.sem_add_columns(web_page_cols, desc="Extract HTML content")
        table_urls = papers_html.sem_add_columns(url_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        output = table_urls

        # urlFile = pz.Dataset("testdata/biofabric-urls", schema=TextFile)
        # table_urls = table_urls.sem_add_columns(URL, desc="The URLs of the tables")
        # tables = table_urls.sem_add_columns(File, udf=udfs.url_to_file)
        # xls = tables.sem_add_columns(XLSFile, udf = udfs.file_to_xls)
        # patient_tables = xls.sem_add_columns(Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        # output = patient_tables

    elif experiment == "filtering":
        xls = pz.Dataset("testdata/biofabric-tiny")
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, cols=table_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.sem_filter("The rows of the table contain the patient age")
        # patient_tables = patient_tables.sem_filter("The table explains the meaning of attributes")
        # patient_tables = patient_tables.sem_filter("The table contains patient biometric data")
        # patient_tables = patient_tables.sem_filter("The table contains proteomic data")
        # patient_tables = patient_tables.sem_filter("The table records if the patient is excluded from the study")
        output = patient_tables

    elif experiment == "matching":
        xls = pz.Dataset("testdata/biofabric-matching")
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, cols=table_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        case_data = patient_tables.sem_add_columns(case_data_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        output = case_data

    elif experiment == "endtoend":
        xls = pz.Dataset("testdata/biofabric-tiny")
        patient_tables = xls.add_columns(udf=udfs.xls_to_tables, cols=table_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.sem_filter("The rows of the table contain the patient age")
        case_data = patient_tables.sem_add_columns(case_data_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        output = case_data

    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=False,
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
