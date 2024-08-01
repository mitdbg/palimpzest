#!/usr/bin/env python3
""" This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""

import palimpzest as pz
import pandas as pd
import time
import os

pz.DataDirectory().clearCache(keep_registry=True)


import networkx as nx
import streamlit as st
import context
import palimpzest as pz
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import requests
import json
import time
import os

class ScientificPaper(pz.PDFFile):
   """Represents a scientific research paper, which in practice is usually from a PDF file"""
   paper_title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   paper_year = pz.Field(desc="The year the paper was published. This is a number.", required=False)
   paper_author = pz.Field(desc="The name of the first author of the paper", required=True)
   paper_journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
   paper_subject = pz.Field(desc="A summary of the paper contribution in one sentence", required=False)
   paper_doiURL = pz.Field(desc="The DOI URL for the paper", required=True)

class Reference(pz.Schema):
    """ Represents a reference to another paper, which is cited in a scientific paper"""
    reference_index = pz.Field(desc="The index of the reference in the paper", required=True)
    reference_title = pz.Field(desc="The title of the paper being cited", required=True)
    reference_first_author = pz.Field(desc="The author of the paper being cited", required=True)
    reference_year = pz.Field(desc="The year in which the cited paper was published", required=True)
    # snippet = pz.Field(desc="A snippet from the source paper that references the index", required=False)


class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    ajcc_pathologic_t = pz.Field(desc="Code of pathological T (primary tumor) to define the size or contiguous extension of the primary tumor (T), using staging criteria from the American Joint Committee on Cancer (AJCC).", required=False)
    ajcc_pathologic_n = pz.Field(desc="The codes that represent the stage of cancer based on the nodes present (N stage) according to criteria based on multiple editions of the AJCC's Cancer Staging Manual.", required=False)
    ajcc_pathologic_stage = pz.Field(desc="The extent of a cancer, especially whether the disease has spread from the original site to other parts of the body based on AJCC staging criteria.", required=False)
    tumor_grade = pz.Field(desc="Numeric value to express the degree of abnormality of cancer cells, a measure of differentiation and aggressiveness.", required=False)
    tumor_focality = pz.Field(desc="The text term used to describe whether the patient's disease originated in a single location or multiple locations.", required=False)
    tumor_largest_dimension_diameter = pz.Field(desc="The tumor largest dimension diameter.", required=False)
    primary_diagnosis = pz.Field(desc="Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).", required=False)
    morphology = pz.Field(desc="The Morphological code of the tumor, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).", required=False)
    tissue_or_organ_of_origin = pz.Field(desc="The text term used to describe the anatomic site of origin, of the patient's malignant disease, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).", required=False)
    # tumor_code = pz.Field(desc="The tumor code", required=False)
    study = pz.Field(desc="The last name of the author of the study, from the table name", required=False)



@st.cache_resource()
def extract_supplemental(engine, policy):
    papers = pz.Dataset("biofabric-pdf", schema=ScientificPaper)
    paperURLs = papers.convert(pz.URL, desc="The DOI url of the paper") 
    htmlDOI = paperURLs.map(pz.DownloadHTMLFunction())
    tableURLS = htmlDOI.convert(pz.URL, desc="The URLs of the XLS tables from the page", cardinality="oneToMany")
    # urlFile = pz.Dataset("biofabric-urls", schema=pz.TextFile)
    # tableURLS = urlFile.convert(pz.URL, desc="The URLs of the tables")
    binary_tables = tableURLS.map(pz.DownloadBinaryFunction())
    tables = binary_tables.convert(pz.File)
    xls = tables.convert(pz.XLSFile)
    patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")

    output = patient_tables
    iterable  =  pz.Execute(patient_tables,
                                    policy = policy,
                                    nocache=True,
                                    allow_code_synth=False,
                                    allow_token_reduction=False,
                                    execution_engine=engine)

    tables = []
    statistics = []
    for table, plan, stats in iterable:
        record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats

@st.cache_resource()
def integrate_tables(engine, policy):
    xls = pz.Dataset('biofabric-tiny', schema=pz.XLSFile)
    patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
    patient_tables = patient_tables.filter("The table contains biometric information about the patient")
    case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

    iterable  =  pz.Execute(case_data,
                                    policy = policy,
                                    nocache=True,
                                    allow_code_synth=False,
                                    allow_token_reduction=False,
                                    execution_engine=engine)

    tables = []
    statistics = []
    for table, plan, stats in iterable:
        record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats

@st.cache_resource()
def extract_references(engine, policy):
    papers = pz.Dataset("bdf-usecase3-tiny", schema=ScientificPaper)
    papers = papers.filter("The paper mentions phosphorylation of Exo1")
    references = papers.convert(Reference, desc="A paper cited in the reference section", cardinality="oneToMany")

    output = references
    iterable  =  pz.Execute(output,
                            policy = policy,
                            nocache=True,
                            allow_sentinels = False,
                            allow_code_synth=False,
                            allow_token_reduction=False,
                            execution_engine=engine)

    tables = []
    statistics = []
    for table, plan, stats in iterable:
        record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats


pdfdir = "testdata/bdf-usecase3-pdf/"

with st.sidebar:
    datasets = pz.DataDirectory().listRegisteredDatasets()
    options = [name for name, path in datasets if path[0] == "dir"]
    options = [name for name in options if "bdf-usecase3" in name]
    dataset = st.radio("Select a dataset", options)
    run_pz = st.button("Run Palimpzest on dataset")

    # st.radio("Biofabric Data Integration")
run_pz = False
dataset = "bdf-usecase3-tiny"

if run_pz:
    # reference, plan, stats = run_workload()
    papers = pz.Dataset(dataset, schema=ScientificPaper)
    papers = papers.filter("The paper mentions phosphorylation of Exo1")
    output = papers.convert(Reference, desc="The references cited in the paper", cardinality="oneToMany")

    # output = references
    # engine = pz.NoSentinelExecution
    engine = pz.StreamingSequentialExecution
    # policy = pz.MinCost()
    policy = pz.MaxQuality()
    iterable  =  pz.Execute(output,
                            policy = policy,
                            nocache=True,
                            allow_sentinels = False,
                            allow_code_synth=False,
                            allow_token_reduction=False,
                            execution_engine=engine)

    references = []
    statistics = []

    for idx, (reference, plan, stats) in enumerate(iterable):
        
        record_time = time.time()
        statistics.append(stats)

        if not idx:
            with st.container():
                st.write("### Executed plan: \n")
                # st.write(" " + str(plan).replace("\n", "  \n "))
                for idx, op in enumerate(plan.operators):
                    strop = f"{idx+1}. {str(op)}"
                    strop = strop.replace("\n", "  \n")
                    st.write(strop)
        for ref in reference:
            try:
                index = ref.index
            except:
                continue
            ref.key = ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
            references.append({
                "title": ref.title,
                "index": index,
                "first_author": ref.first_author,
                "year": ref.year,
                # "snippet": ref.snippet,
                "source": ref.filename,
                "key": ref.key,
            })

            with st.container(height=200, border=True):
                st.write(" **idx:** ", ref.index)
                st.write(" **Paper:** ", ref.title)
                st.write(" **Author:**" ,ref.first_author)
                st.write(" **Year:** ", ref.year)
                st.write(" **Key:** ", ref.key)
                # st.write(" **Reference text:** ", ref.snippet, "\n")
    references_df = pd.DataFrame(references)

else:
    reference_dir = "testdata/bdf-usecase3-references/"
    references = []
    for file in os.listdir(reference_dir):
        df = pd.read_csv(os.path.join(reference_dir, file))
        # create first_title as the first word of the title column
        df["first_title"] = df["title"].apply(lambda x: x.split()[0])
        try:
            df["first_author"] = df["authors"].apply(lambda x: x.split()[0])
        except:
            breakpoint()
        df["key"] = df["first_author"] + df["first_title"] + df["year"].astype(str)
        references.append(df)
    references_df = pd.concat(references)

G = nx.DiGraph()
try:
    G.add_nodes_from(references_df["key"].values)
except:
    breakpoint()
try:
    G.add_nodes_from(references_df["source"].unique())
    for idx, row in references_df.iterrows():
        G.add_edge(row["source"], row["key"])
except:
    G.add_nodes_from(references_df["filename"].unique())
    for idx, row in references_df.iterrows():
        G.add_edge(row["filename"], row["key"])

# prune all nodes with no edges or one edge
pruned_nodes = [node for node in G.nodes if G.degree(node) == 0]
pruned_nodes += [node for node in G.nodes if G.degree(node) == 1]
G.remove_nodes_from(pruned_nodes)

st.title("Graph network")
fig, ax = plt.subplots()
pos = nx.random_layout(G)
nx.draw(G,pos, with_labels=True)
st.pyplot(fig)

nx.write_gexf(G, "demos/bdf-usecase3.gexf")

print("References:", references_df)
# st.write(table.title, table.author, table.abstract)
# endTime = time.time()
# print("Elapsed time:", endTime - startTime)