#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""

import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st  # type: ignore

from palimpzest.constants import Cardinality
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import PDFFile, Schema
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import MaxQuality, MinCost
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory
from palimpzest.sets import Dataset

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class ScientificPaper(PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    paper_title = Field(
        desc="The title of the paper. This is a natural language title, not a number or letter."
    )
    author = Field(desc="The name of the first author of the paper")
    #    publicationYear = Field(desc="The year the paper was published. This is a number.")
    #    journal = Field(desc="The name of the journal the paper was published in")
    abstract = Field(desc="A short description of the paper contributions and findings")


#    doiURL se= Field(desc="The DOI URL for the paper")


class Reference(Schema):
    """Represents a reference to another paper, which is cited in a scientific paper"""

    index = Field(desc="The index of the reference in the paper")
    title = Field(desc="The title of the paper being cited")
    first_author = Field(desc="The author of the paper being cited")
    year = Field(desc="The year in which the cited paper was published")
    # snippet = Field(desc="A snippet from the source paper that references the index")


@st.cache_resource()
def run_workload():
    papers = Dataset("bdf-usecase3-tiny", schema=ScientificPaper)
    # papers = papers.filter("The paper mentions phosphorylation of Exo1")
    references = papers.convert(
        Reference, desc="A paper cited in the reference section", cardinality=Cardinality.ONE_TO_MANY
    )

    output = references
    # engine = NoSentinelExecution
    policy = MinCost()
    config = QueryProcessorConfig(
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        processing_strategy="streaming",
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    iterable = output.run(config)

    tables = []
    statistics = []
    for data_record_collection in iterable:  # noqa: B007
        # record_time = time.time()
        table = data_record_collection.data_records
        stats = data_record_collection.plan_stats
        tables += table
        statistics.append(stats)

    processor = QueryProcessorFactory.create_processor(output, config)
    plan = processor.generate_plan(output, policy)
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
    processor = QueryProcessorFactory.create_processor(output, config)
    plan =processor.generate_plan(output, policy)
    iterable = output.run(config)

    references = []
    statistics = []

    for idx, data_record_collection in enumerate(iterable):
        record_time = time.time()
        references = data_record_collection.data_records
        stats = data_record_collection.plan_stats
        plan = data_record_collection.executed_plans[0]
        statistics.append(stats)

        if not idx:
            with st.container():
                st.write("### Executed plan: \n")
                st.write(" " + str(plan).replace("\n", "  \n "))
                
        for ref in references:
            try:
                index = ref.index
            except Exception:
                continue
            # ref.key = ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
            references.append(
                {
                    "title": ref.title,
                    "index": index,
                    "first_author": ref.first_author,
                    "year": ref.year,
                    # "snippet": ref.snippet,
                    "source": ref.filename,
                    # "key": ref.key,
                }
            )

            with st.container(height=200, border=True):
                st.write(" **idx:** ", ref.index)
                st.write(" **Paper:** ", ref.title)
                st.write(" **Author:**", ref.first_author)
                st.write(" **Year:** ", ref.year)
                # st.write(" **Key:** ", ref.key)
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
# end_time = time.time()
# print("Elapsed time:", end_time - start_time)
