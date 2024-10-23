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
    from palimpzest.utils import load_env

    load_env()


class ScientificPaper(pz.PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    title = pz.Field(
        desc="The title of the paper. This is a natural language title, not a number or letter.",
        required=True,
    )
    publicationYear = pz.Field(
        desc="The year the paper was published. This is a number.",
        required=False,
    )
    author = pz.Field(
        desc="The name of the first author of the paper", required=True
    )
    journal = pz.Field(
        desc="The name of the journal the paper was published in", required=True
    )
    subject = pz.Field(
        desc="A summary of the paper contribution in one sentence",
        required=False,
    )
    doiURL = pz.Field(desc="The DOI URL for the paper", required=True)


class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""

    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(
        desc="The age of the patient at the time of diagnosis", required=False
    )
    race = pz.Field(
        desc="An arbitrary classification of a taxonomic group that is a division of a species.",
        required=False,
    )
    ethnicity = pz.Field(
        desc="Whether an individual describes themselves as Hispanic or Latino or not.",
        required=False,
    )
    gender = pz.Field(
        desc="Text designations that identify gender.", required=False
    )
    vital_status = pz.Field(
        desc="The vital status of the patient", required=False
    )
    ajcc_pathologic_t = pz.Field(desc="The AJCC pathologic T", required=False)
    ajcc_pathologic_n = pz.Field(desc="The AJCC pathologic N", required=False)
    ajcc_pathologic_stage = pz.Field(
        desc="The AJCC pathologic stage", required=False
    )
    tumor_grade = pz.Field(desc="The tumor grade", required=False)
    tumor_focality = pz.Field(desc="The tumor focality", required=False)
    tumor_largest_dimension_diameter = pz.Field(
        desc="The tumor largest dimension diameter", required=False
    )
    primary_diagnosis = pz.Field(desc="The primary diagnosis", required=False)
    morphology = pz.Field(desc="The morphology", required=False)
    tissue_or_organ_of_origin = pz.Field(
        desc="The tissue or organ of origin", required=False
    )
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    study = pz.Field(
        desc="The last name of the author of the study, from the table name",
        required=False,
    )


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
    startTime = time.time()
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--no-cache", action="store_true", help="Do not use cached results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Do not use cached results",
        default=True,
    )
    parser.add_argument(
        "--from_xls",
        action="store_true",
        help="Start from pre-downloaded excel files",
        default=False,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="The experiment to run",
        default="matching",
    )
    parser.add_argument(
        "--policy", type=str, help="The policy to use", default="cost"
    )
    parser.add_argument(
        "--engine", type=str, help="The engine to use", default="parallel"
    )

    args = parser.parse_args()
    no_cache = args.no_cache
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy
    experiment = args.experiment
    engine = args.engine
    if engine == "sequential":
        engine = pz.SequentialSingleThreadSentinelExecution
    elif engine == "parallel":
        engine = pz.PipelinedParallelSentinelExecution
    elif engine == "nosentinel":
        engine = pz.SequentialSingleThreadNoSentinelExecution

    if no_cache:
        pz.DataDirectory().clearCache(keep_registry=True)

    if policy == "cost":
        policy = pz.MinCost()
    elif policy == "quality":
        policy = pz.MaxQuality()
    else:
        policy = pz.UserChoice()

    if experiment == "collection":
        # papers = pz.Dataset("biofabric-pdf", schema=ScientificPaper)
        # paperURLs = papers.convert(pz.URL, desc="The DOI url of the paper")
        # TODO this fetch should be refined to work for all papers
        # htmlDOI = paperURLs.map(pz.DownloadHTMLFunction())
        papers_html = pz.Dataset("biofabric-html", schema=pz.WebPage)
        tableURLS = papers_html.convert(
            pz.URL,
            desc="The URLs of the XLS tables from the page",
            cardinality=pz.Cardinality.ONE_TO_MANY,
        )
        output = tableURLS
        # urlFile = pz.Dataset("biofabric-urls", schema=pz.TextFile)
        # tableURLS = tableURLS.convert(pz.URL, desc="The URLs of the tables")
        # tables = tableURLS.convert(pz.File, udf=udfs.url_to_file)
        # xls = tables.convert(pz.XLSFile, udf = udfs.file_to_xls)
        # patient_tables = xls.convert(pz.Table, udf=udfs.xls_to_tables, cardinality=pz.Cardinality.ONE_TO_MANY)
        # output = patient_tables

    elif experiment == "filtering":
        xls = pz.Dataset("biofabric-tiny", schema=pz.XLSFile)
        patient_tables = xls.convert(
            pz.Table,
            udf=udfs.xls_to_tables,
            cardinality=pz.Cardinality.ONE_TO_MANY,
        )
        patient_tables = patient_tables.filter(
            "The rows of the table contain the patient age"
        )
        # patient_tables = patient_tables.filter("The table explains the meaning of attributes")
        # patient_tables = patient_tables.filter("The table contains patient biometric data")
        # patient_tables = patient_tables.filter("The table contains proteomic data")
        # patient_tables = patient_tables.filter("The table records if the patient is excluded from the study")
        output = patient_tables

    elif experiment == "matching":
        xls = pz.Dataset("biofabric-matching", schema=pz.XLSFile)
        patient_tables = xls.convert(
            pz.Table,
            udf=udfs.xls_to_tables,
            cardinality=pz.Cardinality.ONE_TO_MANY,
        )
        case_data = patient_tables.convert(
            CaseData,
            desc="The patient data in the table",
            cardinality="oneToMany",
        )
        output = case_data

    elif experiment == "endtoend":
        xls = pz.Dataset("biofabric-tiny", schema=pz.XLSFile)
        patient_tables = xls.convert(
            pz.Table,
            udf=udfs.xls_to_tables,
            cardinality=pz.Cardinality.ONE_TO_MANY,
        )
        patient_tables = patient_tables.filter(
            "The rows of the table contain the patient age"
        )
        case_data = patient_tables.convert(
            CaseData,
            desc="The patient data in the table",
            cardinality="oneToMany",
        )
        output = case_data

    tables, plan, stats = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        execution_engine=engine,
    )

    print_table(tables)
    print(plan)
    print(stats)

    endTime = time.time()
    print("Elapsed time:", endTime - startTime)
