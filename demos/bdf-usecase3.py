#!/usr/bin/env python3
""" This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""
from pypdf import PdfReader

import networkx as nx
import streamlit as st
from tqdm import tqdm 
import context
from palimpzest.constants import PZ_DIR
import palimpzest as pz
import pdb 
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
   title = pz.Field(desc="The title of the paper. This is a natural language title, not a number or letter.", required=True)
   author = pz.Field(desc="The name of the first author of the paper", required=True)
#    publicationYear = pz.Field(desc="The year the paper was published. This is a number.", required=False)
#    journal = pz.Field(desc="The name of the journal the paper was published in", required=True)
   abstract = pz.Field(desc="A short description of the paper contributions and findings", required=False)
#    doiURL = pz.Field(desc="The DOI URL for the paper", required=True)

class Reference(pz.Schema):
    """ Represents a reference to another paper, which is cited in a scientific paper"""
    index = pz.Field(desc="The index of the reference in the paper", required=True)
    title = pz.Field(desc="The title of the paper being cited", required=True)
    first_author = pz.Field(desc="The author of the paper being cited", required=True)
    year = pz.Field(desc="The year in which the cited paper was published", required=True)
    snippet = pz.Field(desc="A snippet from the source paper that references the index", required=False)

@st.cache_resource()
def run_workload():
    papers = pz.Dataset("bdf-usecase3-references-pdf", schema=ScientificPaper)
    # papers = papers.filter("The paper mentions phosphorylation of Exo1")
    references = papers.convert(Reference, desc="The references cited in the paper", cardinality="oneToMany")

    output = references
    # engine = pz.NoSentinelExecution
    engine = pz.StreamingSequentialExecution
    policy = pz.MinCost()
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
def reference_graph(references_df):
    G = nx.DiGraph()
    G.add_nodes_from(references_df["key"].values)
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
    return G


run_pz = True
if run_pz:
    # reference, plan, stats = run_workload()

    papers = pz.Dataset("bdf-usecase3-references-pdf", schema=ScientificPaper)
    # papers = papers.filter("The paper mentions phosphorylation of Exo1")
    output = papers.convert(Reference, desc="The references cited in the paper", cardinality="oneToMany")

    # output = references
    # engine = pz.NoSentinelExecution
    engine = pz.StreamingSequentialExecution
    policy = pz.MinCost()
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
        references += reference
        statistics.append(stats)

        if not idx:
            with st.container():
                st.write("### Executed plan: \n")
                # st.write(" " + str(plan).replace("\n", "  \n "))
                for idx, op in enumerate(plan.operators):
                    strop = f"{idx+1}. {str(op)}"
                    strop = strop.replace("\n", "  \n")
                    st.write(strop)
        
        st.write(str(stats))

        for ref in reference:
            try:
                index = ref.index
            except:
                continue
            references.append({
                "title": ref.title,
                "index": index,
                "first_author": ref.first_author,
                "year": ref.year,
                "snippet": ref.snippet,
                "source": ref.filename,
                "key": ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
            })

            with st.container(height=200, border=True):
                st.write(" **idx:** ", ref.index)
                st.write(" **Paper:** ", ref.title)
                st.write(" **Author:**" ,ref.first_author)
                st.write(" **Year:** ", ref.year)
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

G = reference_graph(references_df)
# print(references_df)
print()

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

