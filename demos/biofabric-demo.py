#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
Make sure to run:
python src/cli/cli_main.py reg --path testdata/biofabric-urls/ --name biofabric-urls

"""

import argparse
import os
import time

from palimpzest.constants import Cardinality
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import URL, Schema, Table, WebPage, XLSFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils import udfs

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class CaseData(Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = Field(desc="The ID of the case")
    age_at_diagnosis = Field(desc="The age of the patient at the time of diagnosis")
    race = Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species."
    )
    ethnicity = Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not."
    )
    gender = Field(desc="Text designations that identify gender.")
    vital_status = Field(desc="The vital status of the patient")
    ajcc_pathologic_t = Field(desc="The AJCC pathologic T")
    ajcc_pathologic_n = Field(desc="The AJCC pathologic N")
    ajcc_pathologic_stage = Field(desc="The AJCC pathologic stage")
    tumor_grade = Field(desc="The tumor grade")
    tumor_focality = Field(desc="The tumor focality")
    tumor_largest_dimension_diameter = Field(desc="The tumor largest dimension diameter")
    primary_diagnosis = Field(desc="The primary diagnosis")
    morphology = Field(desc="The morphology")
    tissue_or_organ_of_origin = Field(desc="The tissue or organ of origin")
    # tumor_code = Field(desc="The tumor code")
    study = Field(desc="The last name of the author of the study, from the table name")


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
        papers_html = Dataset("biofabric-html", schema=WebPage)
        table_urls = papers_html.convert(
            URL, desc="The URLs of the XLS tables from the page", cardinality=Cardinality.ONE_TO_MANY
        )
        output = table_urls
        # urlFile = Dataset("biofabric-urls", schema=TextFile)
        # table_urls = table_urls.convert(URL, desc="The URLs of the tables")
        # tables = table_urls.convert(File, udf=udfs.url_to_file)
        # xls = tables.convert(XLSFile, udf = udfs.file_to_xls)
        # patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        # output = patient_tables

    elif experiment == "filtering":
        xls = Dataset("biofabric-tiny", schema=XLSFile)
        patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.filter("The rows of the table contain the patient age")
        # patient_tables = patient_tables.filter("The table explains the meaning of attributes")
        # patient_tables = patient_tables.filter("The table contains patient biometric data")
        # patient_tables = patient_tables.filter("The table contains proteomic data")
        # patient_tables = patient_tables.filter("The table records if the patient is excluded from the study")
        output = patient_tables

    elif experiment == "matching":
        xls = Dataset("biofabric-matching", schema=XLSFile)
        patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        case_data = patient_tables.convert(
            CaseData, desc="The patient data in the table", cardinality=Cardinality.ONE_TO_MANY
        )
        output = case_data

    elif experiment == "endtoend":
        xls = Dataset("biofabric-tiny", schema=XLSFile)
        patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.filter("The rows of the table contain the patient age")
        case_data = patient_tables.convert(
            CaseData, desc="The patient data in the table", cardinality=Cardinality.ONE_TO_MANY
        )
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

    print_table(data_record_collection.data_records)
    print(data_record_collection.executed_plans)
    # print(data_record_collection.execution_stats)

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
