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


class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False
    )
    ethnicity = pz.Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False
    )
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The AJCC pathologic stage", required=False)
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(desc="The tumor largest dimension diameter", required=False)
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(desc="The tissue or organ of origin", required=False)
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    study = pz.Field(desc="The last name of the author of the study, from the table name", required=False)


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
    parser.add_argument("--executor", type=str, help="The plan executor to use", default="parallel")

    args = parser.parse_args()
    no_cache = args.no_cache
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy
    experiment = args.experiment
    executor = args.executor
    execution_engine = None
    if executor == "sequential":
        execution_engine = pz.NoSentinelSequentialSingleThreadExecution
    elif executor == "pipelined":
        execution_engine = pz.NoSentinelPipelinedSingleThreadExecution
    elif executor == "parallel":
        execution_engine = pz.NoSentinelPipelinedParallelExecution
    else:
        print("Executor not supported for this demo")
        exit(1)

    if no_cache:
        pz.DataDirectory().clear_cache(keep_registry=True)

    if policy == "cost":
        policy = pz.MinCost()
    elif policy == "quality":
        policy = pz.MaxQuality()

    if experiment == "collection":
        papers_html = pz.Dataset("biofabric-html", schema=pz.WebPage)
        table_urls = papers_html.convert(
            pz.URL, desc="The URLs of the XLS tables from the page", cardinality=pz.Cardinality.ONE_TO_MANY
        )
        output = table_urls
        # urlFile = pz.Dataset("biofabric-urls", schema=pz.TextFile)
        # table_urls = table_urls.convert(pz.URL, desc="The URLs of the tables")
        # tables = table_urls.convert(pz.File, udf=udfs.url_to_file)
        # xls = tables.convert(pz.XLSFile, udf = udfs.file_to_xls)
        # patient_tables = xls.convert(pz.Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        # output = patient_tables

    elif experiment == "filtering":
        xls = pz.Dataset("biofabric-tiny", schema=pz.XLSFile)
        patient_tables = xls.convert(pz.Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.filter("The rows of the table contain the patient age")
        # patient_tables = patient_tables.filter("The table explains the meaning of attributes")
        # patient_tables = patient_tables.filter("The table contains patient biometric data")
        # patient_tables = patient_tables.filter("The table contains proteomic data")
        # patient_tables = patient_tables.filter("The table records if the patient is excluded from the study")
        output = patient_tables

    elif experiment == "matching":
        xls = pz.Dataset("biofabric-matching", schema=pz.XLSFile)
        patient_tables = xls.convert(pz.Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        case_data = patient_tables.convert(
            CaseData, desc="The patient data in the table", cardinality=pz.Cardinality.ONE_TO_MANY
        )
        output = case_data

    elif experiment == "endtoend":
        xls = pz.Dataset("biofabric-tiny", schema=pz.XLSFile)
        patient_tables = xls.convert(pz.Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        patient_tables = patient_tables.filter("The rows of the table contain the patient age")
        case_data = patient_tables.convert(
            CaseData, desc="The patient data in the table", cardinality=pz.Cardinality.ONE_TO_MANY
        )
        output = case_data

    tables, plan, stats = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        execution_engine=execution_engine,
    )

    print_table(tables)
    print(plan)
    print(stats)

    end_time = time.time()
    print("Elapsed time:", end_time - start_time)
