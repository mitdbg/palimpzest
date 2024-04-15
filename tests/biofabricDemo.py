#!/usr/bin/env python3
""" This scripts is a demo for the biofabric data integration.
Make sure to run:
python src/cli/cli_main.py reg --path testdata/biofabric-urls/ --name biofabric-urls

"""
import context
from palimpzest.constants import PZ_DIR
import palimpzest as pz

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

if __name__ == "__main__":
    startTime = time.time()
    parser = argparse.ArgumentParser(description='Run a simple demo')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached results')
    parser.add_argument('--verbose', action='store_true', help='Do not use cached results', default=True)
    parser.add_argument('--from_xls', action='store_true', help='Start from pre-downloaded excel files', default=False)
    parser.add_argument('--policy', type=str, help='The policy to use', default='cost')

    args = parser.parse_args()
    no_cache = args.no_cache
    verbose = args.verbose
    from_xls = args.from_xls
    policy = args.policy

    if no_cache:
        pz.DataDirectory().clearCache(keep_registry=True)

    if policy == 'cost':
        policy = pz.MinCost()
    elif policy == 'quality':
        policy = pz.MaxQuality()
    else:
        policy = pz.UserChoice()

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

    if from_xls:
        xls = pz.Dataset('biofabric-xls', schema=pz.XLSFile)
        
    patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    # patient_tables = patient_tables.filterByStr("The table explains the meaning of attributes")
    # patient_tables = patient_tables.filterByStr("The table records if the patient is excluded from the study")
    # patient_tables = patient_tables.filterByStr("The table contains patient biometric data")
    # patient_tables = patient_tables.filterByStr("The table contains proteomic data")

    output = patient_tables
    execution = pz.SimpleExecution(output, policy)
    physicalTree = execution.executeAndOptimize(verbose=args.verbose)

    for table in physicalTree:
        header = table.header
        subset_rows = table.rows[:3]

        print("Table name:", table.name)
        print(" | ".join(header)[:100], "...")
        for row in subset_rows:
            print(" | ".join(row.cells)[:100], "...")
        print()


    endTime = time.time()
    print("Elapsed time:", endTime - startTime)

