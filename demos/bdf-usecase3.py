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

import palimpzest as pz

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils import load_env

    load_env()


class ScientificPaper(pz.PDFFile):
    """Represents a scientific research paper, which in practice is usually from a PDF file"""

    paper_title = pz.Field(
        desc="The title of the paper. This is a natural language title, not a number or letter.",
        required=True,
    )
    author = pz.Field(desc="The name of the first author of the paper", required=True)
    #    publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
    #    journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
    abstract = pz.Field(
        desc="A short description of the paper contributions and findings",
        required=False,
    )


#    doiURL se= pz.Field(desc="The DOI URL for the paper", required=True)


class Reference(pz.Schema):
    """Represents a reference to another paper, which is cited in a scientific paper"""

    index = pz.Field(desc="The index of the reference in the paper", required=True)
    title = pz.Field(desc="The title of the paper being cited", required=True)
    first_author = pz.Field(desc="The author of the paper being cited", required=True)
    year = pz.Field(desc="The year in which the cited paper was published", required=True)
    # snippet = pz.Field(desc="A snippet from the source paper that references the index", required=False)


@st.cache_resource()
def run_workload():
    papers = pz.Dataset("bdf-usecase3-tiny", schema=ScientificPaper)
    # papers = papers.filter("The paper mentions phosphorylation of Exo1")
    references = papers.convert(
        Reference,
        desc="A paper cited in the reference section",
        cardinality=pz.Cardinality.ONE_TO_MANY,
    )

    output = references
    # engine = pz.NoSentinelExecution
    engine = pz.StreamingSequentialExecution
    policy = pz.MinCost()
    iterable = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        execution_engine=engine,
    )

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
run_pz = True
dataset = "bdf-usecase3-tiny"

if run_pz:
    # reference, plan, stats = run_workload()
    papers = pz.Dataset(dataset, schema=ScientificPaper)
    papers = papers.filter("The paper mentions phosphorylation of Exo1")
    output = papers.convert(
        Reference,
        desc="The references cited in the paper",
        cardinality="oneToMany",
    )

    # output = references
    # engine = pz.NoSentinelExecution
    engine = pz.StreamingSequentialExecution
    # policy = pz.MinCost()
    policy = pz.MaxQuality()
    iterable = pz.Execute(
        output,
        policy=policy,
        nocache=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        execution_engine=engine,
    )

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
nx.draw(G, pos, with_labels=True)
st.pyplot(fig)

nx.write_gexf(G, "demos/bdf-usecase3.gexf")

print("References:", references_df)
# st.write(table.title, table.author, table.abstract)
# endTime = time.time()
# print("Elapsed time:", endTime - startTime)
