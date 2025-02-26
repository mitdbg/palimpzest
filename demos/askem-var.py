#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""

import json
import time

import pandas as pd
import streamlit as st

import palimpzest as pz
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory

dict_of_excerpts = [
    {"id": 0, "text": "ne of the few states producing detailed daily reports of COVID-19 confirmed cases, COVID-19 related cumulative hospitalizations, intensive care unit (ICU) admissions, and deaths per county. Likewise, Ohio is a state with marked variation of demographic and geographic attributes among counties along with substantial differences in the capacity of healthcare within the state. Our aim is to predict the spatiotemporal dynamics of the COVID-19 pandemic in relation with the distribution of the capacity of healthcare in Ohio. 2. Methods 2.1. Mathematical model We developed a spatial mathematical model to simulate the transmission dynamics of COVID-19 disease infection and spread. The spatially-explicit model incorporates geographic connectivity information at county level. The Susceptible-Infected-Hospitalized-Recovered- Dead (SIHRD) COVID-19 model classified the population into susceptibles (S), confirmed infections (I), hospitalized and ICU admitted (H), recovered (R) and dead (D). Based on a previous study that identified local air hubs and main roads as important geospatial attributes lio residing in the county. In the second scenario, we used the model to generate projections of the impact of potential easing on the non-pharmaceutical interventions in the critical care capacity of each county in Ohio. We assessed the impact of 50% reduction on the estimated impact of non-pharmaceutical interventions in reducing the hazard rate of infection. Under this scenario we calculated the proportion of ICU \n'"},
    {"id": 1, "text": "t model incorporates geographic connectivity information at county level. The Susceptible-Infected-Hospitalized-Recovered- Dead (SIHRD) COVID-19 model classified the population into susceptibles (S), confirmed infections (I), hospitalized and ICU admitted (H), recovered (R) and dead (D). Based on a previous study that identified local air hubs and main roads as important geospatial attributes linked to differential COVID-19 related hospitalizations and mortality (Correa-Agudelo et a"}
]

variable_cols = [
    {"name": "name", "type": str, "desc": "The label used for a the scientific variable, like a, b, 𝜆 or 𝜖, NOT None"},
    {"name": "description", "type": str, "desc": "A description of the variable, optional, set 'null' if not found"},
    {"name": "value", "type": int | float, "desc": "The value of the variable, optional, set 'null' if not found"}
]

list_of_strings = ["I have a variable a, the value is 1", "I have a variable b, the value is 2"]
list_of_numbers = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    run_pz = True
    dataset = "askem-tiny"
    file_path = "testdata/askem-tiny/"

    if run_pz:
        dataset = pz.Dataset(pd.DataFrame(list_of_strings))
        dataset = dataset.sem_add_columns(variable_cols, cardinality=pz.Cardinality.ONE_TO_MANY)
        dataset = dataset.sem_filter("The value name is 'a'", depends_on="name")

        policy = pz.MaxQuality()
        config = pz.QueryProcessorConfig(
            policy=policy,
            cache=False,
            verbose=True,
            processing_strategy="streaming",
            execution_strategy="sequential",
            optimizer_strategy="pareto",
        )
        processor = QueryProcessorFactory.create_processor(dataset, config)
        plan = processor.generate_plan(dataset, policy)
        print(processor.plan)

        with st.container():
            st.write("### Executed plan: \n")
            # st.write(" " + str(plan).replace("\n", "  \n "))
            for idx, op in enumerate(plan.operators):
                strop = f"{idx + 1}. {str(op)}"
                strop = strop.replace("\n", "  \n")
                st.write(strop)

        input_records = processor.get_input_records()
        input_df = DataRecord.to_df(input_records)
        print(input_df)

        variables = []
        statistics = []
        # for idx, (vars, plan, stats) in enumerate(iterable):
        for idx, record in enumerate(input_records):
            print(f"idx: {idx}\n record: {record}")
            index = idx
            vars = processor.execute_opstream(processor.plan, record)
            if idx == len(input_records) - 1:
                processor.plan_stats.finish()

            record_time = time.time()
            statistics.append(processor.plan_stats)

            for var in vars:
                # ref.key = ref.first_author.split()[0] + ref.title.split()[0] + str(ref.year)
                try:
                    # set name to None if not found
                    if var.name is None:
                        var.name = "null"
                    # set description to None if not found
                    if var.description is None:
                        var.description = "null"
                    # set value to None if not found
                    if var.value is None:
                        var.value = "null"
                except Exception:
                    continue
                variables.append(
                    {
                        "id": index,
                        "name": var.name,
                        "description": var.description,
                        "value": var.value,
                    }
                )

                # write variables into json file with readable format for every 10 variables
                if index % 10 == 0:
                    with open(f"askem-variables-{dataset}.json", "w") as f:
                        json.dump(variables, f, indent=4)

                with st.container(height=200, border=True):
                    st.write("**id:** ", index)
                    st.write(" **name:** ", var.name)
                    st.write(" **description:** ", var.description)
                    st.write(" **value:** ", var.value, "\n")

        # write variables to a json file with readable format
        with open(f"askem-variables-{dataset}.json", "w") as f:
            json.dump(variables, f, indent=4)
        vars_df = pd.DataFrame(variables)

    # G = nx.DiGraph()
    # try:
    #     G.add_nodes_from(references_df["key"].values)
    # except Exception:
    #     breakpoint()
    # try:
    #     G.add_nodes_from(references_df["source"].unique())
    #     for idx, row in references_df.iterrows():
    #         G.add_edge(row["source"], row["key"])
    # except Exception:
    #     G.add_nodes_from(references_df["filename"].unique())
    #     for idx, row in references_df.iterrows():
    #         G.add_edge(row["filename"], row["key"])
    #
    # # prune all nodes with no edges or one edge
    # pruned_nodes = [node for node in G.nodes if G.degree(node) == 0]
    # pruned_nodes += [node for node in G.nodes if G.degree(node) == 1]
    # G.remove_nodes_from(pruned_nodes)
    #
    # st.title("Graph network")
    # fig, ax = plt.subplots()
    # pos = nx.random_layout(G)
    # nx.draw(G,pos, with_labels=True)
    # st.pyplot(fig)
    #
    # nx.write_gexf(G, "demos/bdf-usecase3.gexf")

    # print("References:", vars_df)
    # st.write(table.title, table.author, table.abstract)
    # endTime = time.time()
    # print("Elapsed time:", endTime - startTime)
