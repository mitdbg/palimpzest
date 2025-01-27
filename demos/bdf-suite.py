#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from palimpzest.constants import Cardinality
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import URL, File, PDFFile, Schema, Table, XLSFile
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset
from palimpzest.utils import udfs

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()

DataDirectory().clear_cache(keep_registry=True)


class ScientificPaper(PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    paper_title = Field(
        desc="The title of the paper. This is a natural language title, not a number or letter."
    )
    paper_year = Field(desc="The year the paper was published. This is a number.")
    paper_author = Field(desc="The name of the first author of the paper")
    paper_journal = Field(desc="The name of the journal the paper was published in")
    paper_subject = Field(desc="A summary of the paper contribution in one sentence")
    paper_doi_url = Field(desc="The DOI URL for the paper")


class Reference(Schema):
    """Represents a reference to another paper, which is cited in a scientific paper"""

    reference_index = Field(desc="The index of the reference in the paper")
    reference_title = Field(desc="The title of the paper being cited")
    reference_first_author = Field(desc="The author of the paper being cited")
    reference_year = Field(desc="The year in which the cited paper was published")
    # snippet = Field(desc="A snippet from the source paper that references the index")


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
    ajcc_pathologic_t = Field(
        desc="Code of pathological T (primary tumor) to define the size or contiguous extension of the primary tumor (T), using staging criteria from the American Joint Committee on Cancer (AJCC).",
    )
    ajcc_pathologic_n = Field(
        desc="The codes that represent the stage of cancer based on the nodes present (N stage) according to criteria based on multiple editions of the AJCC's Cancer Staging Manual.",
    )
    ajcc_pathologic_stage = Field(
        desc="The extent of a cancer, especially whether the disease has spread from the original site to other parts of the body based on AJCC staging criteria.",
    )
    tumor_grade = Field(
        desc="Numeric value to express the degree of abnormality of cancer cells, a measure of differentiation and aggressiveness.",
    )
    tumor_focality = Field(
        desc="The text term used to describe whether the patient's disease originated in a single location or multiple locations.",
    )
    tumor_largest_dimension_diameter = Field(desc="The tumor largest dimension diameter.")
    primary_diagnosis = Field(
        desc="Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).",
    )
    morphology = Field(
        desc="The Morphological code of the tumor, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).",
    )
    tissue_or_organ_of_origin = Field(
        desc="The text term used to describe the anatomic site of origin, of the patient's malignant disease, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).",
    )
    # tumor_code = Field(desc="The tumor code")
    study = Field(desc="The last name of the author of the study, from the table name")


@st.cache_resource()
def extract_supplemental(processing_strategy, execution_strategy, optimizer_strategy, policy):
    papers = Dataset("biofabric-pdf", schema=ScientificPaper)
    paper_urls = papers.convert(URL, desc="The DOI url of the paper")
    html_doi = paper_urls.convert(File, udf=udfs.url_to_file)
    table_urls = html_doi.convert(
        URL, desc="The URLs of the XLS tables from the page", cardinality=Cardinality.ONE_TO_MANY
    )
    # url_file = Dataset("biofabric-urls", schema=TextFile)
    # table_urls = url_file.convert(URL, desc="The URLs of the tables")
    tables = table_urls.convert(File, udf=udfs.url_to_file)
    xls = tables.convert(XLSFile, udf=udfs.file_to_xls)
    patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)

    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy=processing_strategy,
        execution_strategy=execution_strategy,
        optimizer_strategy=optimizer_strategy,
    )
    iterable = patient_tables.run(config)


    tables = []
    statistics = []
    for table, plan, stats in iterable:  # noqa: B007
        # record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats


@st.cache_resource()
def integrate_tables(processing_strategy, execution_strategy, optimizer_strategy, policy):
    xls = Dataset("biofabric-tiny", schema=XLSFile)
    patient_tables = xls.convert(Table, udf=udfs.xls_to_tables, cardinality=Cardinality.ONE_TO_MANY)
    patient_tables = patient_tables.filter("The table contains biometric information about the patient")
    case_data = patient_tables.convert(
        CaseData, desc="The patient data in the table", cardinality=Cardinality.ONE_TO_MANY
    )

    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy=processing_strategy,
        execution_strategy=execution_strategy,
        optimizer_strategy=optimizer_strategy,
    )
    iterable = case_data.run(config)

    tables = []
    statistics = []
    for table, plan, stats in iterable:  # noqa: B007
        # record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats


@st.cache_resource()
def extract_references(processing_strategy, execution_strategy, optimizer_strategy, policy):
    papers = Dataset("bdf-usecase3-tiny", schema=ScientificPaper)
    papers = papers.filter("The paper mentions phosphorylation of Exo1")
    references = papers.convert(
        Reference, desc="A paper cited in the reference section", cardinality=Cardinality.ONE_TO_MANY
    )

    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy=processing_strategy,
        execution_strategy=execution_strategy,
        optimizer_strategy=optimizer_strategy,
    )
    iterable = references.run(config)

    tables = []
    statistics = []
    for table, plan, stats in iterable:  # noqa: B007
        # record_time = time.time()
        tables += table
        statistics.append(stats)

    return tables, plan, stats


pdfdir = "testdata/bdf-usecase3-pdf/"

with st.sidebar:
    datasets = DataDirectory().list_registered_datasets()
    options = [name for name, path in datasets if path[0] == "dir"]
    options = [name for name in options if "bdf-usecase3" in name]
    dataset = st.radio("Select a dataset", options)
    run_pz = st.button("Run Palimpzest on dataset")

    # st.radio("Biofabric Data Integration")
run_pz = True
dataset = "bdf-usecase3-tiny"

if run_pz:
    # reference, plan, stats = run_workload()
    papers = Dataset(dataset, schema=ScientificPaper)
    papers = papers.filter("The paper mentions phosphorylation of Exo1")
    output = papers.convert(Reference, desc="The references cited in the paper", cardinality=Cardinality.ONE_TO_MANY)

    # output = references
    # engine = NoSentinelExecution
    # policy = MinCost()
    policy = MaxQuality()
    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy="streaming",
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    data_record_collection = output.run(config)
    
    references = []
    statistics = []

    for idx, record_collection in enumerate(data_record_collection):
        record_time = time.time()
        stats = record_collection.plan_stats
        references = record_collection.data_records
        plan = record_collection.executed_plans[0]
        statistics.append(stats)

        if not idx:
            with st.container():
                st.write("### Executed plan: \n")
                st.write(" " + str(plan).replace("\n", "  \n "))
                # for idx, op in enumerate(stats.plan_strs[0].operators):
                #     strop = f"{idx+1}. {str(op)}"
                #     strop = strop.replace("\n", "  \n")
                #     st.write(strop)
        for ref in references:
            try:
                index = ref.index
            except Exception:
                continue
            ref.key = ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
            references.append(
                {
                    "title": ref.title,
                    "index": index,
                    "first_author": ref.first_author,
                    "year": ref.year,
                    # "snippet": ref.snippet,
                    "source": ref.filename,
                    "key": ref.key,
                }
            )

            with st.container(height=200, border=True):
                st.write(" **idx:** ", ref.index)
                st.write(" **Paper:** ", ref.title)
                st.write(" **Author:**", ref.first_author)
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
        except Exception:
            breakpoint()
        df["key"] = df["first_author"] + df["first_title"] + df["year"].astype(str)
        references.append(df)
    references_df = pd.concat(references)

G = nx.DiGraph()
try:
    G.add_nodes_from(references_df["key"].values)
except Exception:
    breakpoint()
try:
    G.add_nodes_from(references_df["source"].unique())
    for _, row in references_df.iterrows():
        G.add_edge(row["source"], row["key"])
except Exception:
    G.add_nodes_from(references_df["filename"].unique())
    for _, row in references_df.iterrows():
        G.add_edge(row["filename"], row["key"])

# prune all nodes with no edges or one edge
pruned_nodes = [node for node in G.nodes if G.degree(node) == 0]
pruned_nodes += [node for node in G.nodes if G.degree(node) == 1]
G.remove_nodes_from(pruned_nodes)

st.title("Graph network")
fig, ax = plt.subplots()
pos = nx.random_layout(G)
nx.draw(G, pos, with_labels=True)
st.pyplot(fig)

nx.write_gexf(G, "demos/bdf-usecase3.gexf")

print("References:", references_df)
# st.write(table.title, table.author, table.abstract)
# endTime = time.time()
# print("Elapsed time:", endTime - startTime)
