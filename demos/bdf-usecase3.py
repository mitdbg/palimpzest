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

import palimpzest as pz
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


sci_paper_cols = [
    {"name": "paper_title", "type": str, "desc": "The title of the paper. This is a natural language title, not a number or letter."},
    {"name": "paper_year", "type": int, "desc": "The year the paper was published. This is a number."},
    {"name": "paper_author", "type": str, "desc": "The name of the first author of the paper"},
    {"name": "paper_abstract", "type": str, "desc": "A short description of the paper contributions and findings"},
    {"name": "paper_journal", "type": str, "desc": "The name of the journal the paper was published in"},
    {"name": "paper_subject", "type": str, "desc": "A summary of the paper contribution in one sentence"},
    {"name": "paper_doi_url", "type": str, "desc": "The DOI URL for the paper"}
]

reference_cols = [
    {"name": "reference_index", "type": int | float, "desc": "The index of the reference in the paper"},
    {"name": "reference_title", "type": str, "desc": "The title of the paper being cited"},
    {"name": "reference_first_author", "type": str, "desc": "The author of the paper being cited"},
    {"name": "reference_year", "type": int, "desc": "The year in which the cited paper was published"},
]


@st.cache_resource()
def run_workload():
    papers = pz.Dataset("testdata/bdf-usecase3-tiny")
    papers = papers.sem_add_columns(sci_paper_cols)
    # papers = papers.sem_filter("The paper mentions phosphorylation of Exo1")
    references = papers.sem_add_columns(reference_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
    output = references

    policy = pz.MinCost()
    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=False,
        allow_code_synth=False,
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


with st.sidebar:
    options = [os.path.join("testdata", name) for name in os.listdir("testdata")]
    options = [path for path in options if os.path.isdir(path)]
    options = [path for path in options if "bdf-usecase3" in path]
    dataset = st.radio("Select a dataset", options)
    run_pz = st.button("Run Palimpzest on dataset")

    # st.radio("Biofabric Data Integration")
run_pz = True
dataset = "testdata/bdf-usecase3-tiny"

if run_pz:
    # reference, plan, stats = run_workload()
    papers = pz.Dataset(dataset)
    papers = papers.sem_add_columns(sci_paper_cols)
    papers = papers.sem_filter("The paper mentions phosphorylation of Exo1")
    output = papers.sem_add_columns(reference_cols, cardinality=pz.Cardinality.ONE_TO_MANY)

    # output = references
    # engine = NoSentinelExecution
    # policy = MinCost()
    policy = pz.MaxQuality()
    config = pz.QueryProcessorConfig(
        policy=policy,
        cache=False,
        allow_code_synth=False,
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
        records = data_record_collection.data_records
        stats = data_record_collection.plan_stats
        plan = data_record_collection.executed_plans[0]
        statistics.append(stats)

        if not idx:
            with st.container():
                st.write("### Executed plan: \n")
                st.write(" " + str(plan).replace("\n", "  \n "))
                
        for ref in records:
            try:
                index = ref.reference_index
            except Exception:
                continue
            # ref.key = ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
            references.append(
                {
                    "title": ref.reference_title,
                    "index": index,
                    "first_author": ref.reference_first_author,
                    "year": ref.reference_year,
                    # "snippet": ref.snippet,
                    "source": ref.filename,
                    # "key": ref.key,
                }
            )

            with st.container(height=200, border=True):
                st.write(" **idx:** ", ref.reference_index)
                st.write(" **Paper:** ", ref.reference_title)
                st.write(" **Author:**", ref.reference_first_author)
                st.write(" **Year:** ", ref.reference_year)
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
