#!/usr/bin/env python3
""" This scripts is a demo for the biofabric data integration.
Make sure to run:
python src/cli/cli_main.py reg --path testdata/biofabric-urls/ --name biofabric-urls

"""
import context
from palimpzest.constants import PZ_DIR
import palimpzest as pz
import pdb 
import gradio as gr
import numpy as np
import pandas as pd

import argparse
import requests
import json
import time
import os


class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
   author = pz.Field(desc="The name of the first author of the paper", required=True)
   journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
   subject = pz.Field(desc="A summary of the paper contribution in one sentence", required=False)
   doiURL = pz.Field(desc="The DOI URL for the paper", required=True)

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient in days at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
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


def filtering(input_dataset):
    patient_tables = input_dataset.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    patient_tables = patient_tables.filterByStr("The rows of the table contain the patient age")
    # patient_tables = patient_tables.filterByStr("The table explains the meaning of attributes")
    # patient_tables = patient_tables.filterByStr("The table contains patient biometric data")
    # patient_tables = patient_tables.filterByStr("The table contains proteomic data")
    # patient_tables = patient_tables.filterByStr("The table records if the patient is excluded from the study")
    output = pz.SimpleExecution(patient_tables, policy)
    output = output.executeAndOptimize(args.verbose)

    filtered = []
    for table in output:
        header = table.header
        subset_rows = table.rows[:3]

        print("Table name:", table.name)
        print(" | ".join(header)[:100], "...")
        for row in subset_rows:
            print(" | ".join(row)[:100], "...")
        print()
        
        filtered.append(table.name)

    return filtered, output


def matching(input_dataset):
    patient_tables = input_dataset.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")
    
    matched_tables = pz.SimpleExecution(case_data, policy)   
    matched_tables = matched_tables.executeAndOptimize(args.verbose)
    
    output_rows = []
    for output_table in matched_tables:
        # print([k+":"+str(v)+"\n" for k,v in output_table.asDict().items()])
        # print("------------------------------")
        output_rows.append(output_table.asDict()) 

    output_df = pd.DataFrame(output_rows)
    return output_df, matched_tables

if __name__ == "__main__":
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results')
    parser.add_argument('--verbose', action='store_true', help='Do not use cached results', default=True)
    parser.add_argument('--from_xls', action='store_true', help='Start from pre-downloaded excel files', default=False)
    parser.add_argument('--experiment', type=str, help='The experiment to run', default='matching')
    parser.add_argument('--policy', type=str, help='The policy to use', default='cost')

    args = parser.parse_args()
    no_cache = args.no_cache
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy
    experiment = args.experiment

    if no_cache:
        pz.DataDirectory().clearCache(keep_registry=True)

    if policy == 'cost':
        policy = pz.MinCost()
    elif policy == 'quality':
        policy = pz.MaxQuality()
    else:
        policy = pz.UserChoice()

    if experiment == 'collection':
        papers = pz.Dataset("biofabric-pdf", schema=ScientificPaper)
        paperURLs = papers.convert(pz.URL, desc="The DOI url of the paper") 

        # TODO this fetch should be refined to work for all papers
        htmlDOI = paperURLs.map(pz.DownloadHTMLFunction())
        tableURLS = htmlDOI.convert(pz.URL, desc="The URLs of the XLS tables from the page", cardinality="oneToMany")
        urlFile = pz.Dataset("biofabric-urls", schema=pz.TextFile)
        tableURLS = urlFile.convert(pz.URL, desc="The URLs of the tables")
        binary_tables = tableURLS.map(pz.DownloadBinaryFunction())
        tables = binary_tables.convert(pz.File)
        xls = tables.convert(pz.XLSFile)


    elif experiment == 'filtering':
        xls = pz.Dataset('biofabric-xls', schema=pz.XLSFile)       
        filtered_tables, _ = filtering(xls)
        
        with open("results/biofabric/filtered_tables.txt", "w") as f:
            for item in filtered_tables:
                f.write("%s\n" % item)
        print("Filtered tables:" ,"\n".join(filtered_tables))

    elif experiment == 'matching':
        xls = pz.Dataset('biofabric-matching', schema=pz.XLSFile)
            
        output_df, matched_tables = matching(xls)

        output_df, _ = matching(xls)
        print("Matched table:", output_df)
        out_path = "results/biofabric/"
        output_df.to_csv(out_path+"matched.csv", index=False)
    
    elif experiment == "endtoend":
        xls = pz.Dataset('biofabric-xls', schema=pz.XLSFile)
        filtered_tables, output = filtering(xls)
        output_df, matched_tables = matching(output)

        with open("results/biofabric/filtered_tables_endtoend.txt", "w") as f:
            for item in filtered_tables:
                f.write("%s\n" % item)

        out_path = "results/biofabric/"
        output_df.to_csv(out_path+"matched_endtoend.csv", index=False)

        print("Filtered tables:" ,"\n".join(filtered_tables))
        print("Matched table:", output_df)


    endTime = time.time()
    print("Elapsed time:", endTime - startTime)

