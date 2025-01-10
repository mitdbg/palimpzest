#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""

import json
import time

import pandas as pd
import streamlit as st

import palimpzest as pz
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import Schema
from palimpzest.core.lib.schemas import TextFile
from palimpzest.sets import Dataset
from palimpzest.query import StreamingSequentialExecution
from palimpzest.policy import MaxQuality
from palimpzest.core.elements.records import DataRecord

class Papersnippet(TextFile):
    """Represents an excerpt from a scientific research paper, which potentially contains variables"""

    excerptid = Field(desc="The unique identifier for the excerpt", required=True)
    excerpt = Field(desc="The text of the excerpt", required=True)


class Variable(Schema):
    """Represents a variable of scientific model in a scientific paper"""
    name = Field(desc="The label used for a the scientific variable, like a, b, 𝜆 or 𝜖, NOT None", required=True)
    description = Field(desc="A description of the variable, optional, set 'null' if not found", required=False)
    value = Field(desc="The value of the variable, optional, set 'null' if not found", required=False)


dict_of_excerpts = [
    {"id": 0, "text": "ne of the few states producing detailed daily reports of COVID-19 confirmed cases, COVID-19 related cumulative hospitalizations, intensive care unit (ICU) admissions, and deaths per county. Likewise, Ohio is a state with marked variation of demographic and geographic attributes among counties along with substantial differences in the capacity of healthcare within the state. Our aim is to predict the spatiotemporal dynamics of the COVID-19 pandemic in relation with the distribution of the capacity of healthcare in Ohio. 2. Methods 2.1. Mathematical model We developed a spatial mathematical model to simulate the transmission dynamics of COVID-19 disease infection and spread. The spatially-explicit model incorporates geographic connectivity information at county level. The Susceptible-Infected-Hospitalized-Recovered- Dead (SIHRD) COVID-19 model classified the population into susceptibles (S), confirmed infections (I), hospitalized and ICU admitted (H), recovered (R) and dead (D). Based on a previous study that identified local air hubs and main roads as important geospatial attributes lio residing in the county. In the second scenario, we used the model to generate projections of the impact of potential easing on the non-pharmaceutical interventions in the critical care capacity of each county in Ohio. We assessed the impact of 50% reduction on the estimated impact of non-pharmaceutical interventions in reducing the hazard rate of infection. Under this scenario we calculated the proportion of ICU \n'"},
    {"id": 1, "text": "t model incorporates geographic connectivity information at county level. The Susceptible-Infected-Hospitalized-Recovered- Dead (SIHRD) COVID-19 model classified the population into susceptibles (S), confirmed infections (I), hospitalized and ICU admitted (H), recovered (R) and dead (D). Based on a previous study that identified local air hubs and main roads as important geospatial attributes linked to differential COVID-19 related hospitalizations and mortality (Correa-Agudelo et a"}
]

list_of_strings = ["I have a variable a, the value is 1", "I have a variable b, the value is 2"]
list_of_numbers = [1, 2, 3, 4, 5]

if __name__ == "__main__":
    run_pz = True
    dataset = "askem"
    file_path = "testdata/askem-tiny/"

    if run_pz:
        # reference, plan, stats = run_workload()
        df_input = pd.DataFrame(dict_of_excerpts)
        excerpts = Dataset(df_input)
        output = excerpts.convert(
            Variable, desc="A variable used or introduced in the context", cardinality=pz.Cardinality.ONE_TO_MANY
        ).filter("The value name is 'a'", depends_on="name")
        # policy = pz.MinCost()
        policy = MaxQuality()
        # iterable  =  pz.Execute(output,
        #                         policy = policy,
        #                         nocache=True,
        #                         verbose=True,
        #                         allow_code_synth=False,
        #                         allow_token_reduction=False,
        #                         allow_bonded_query=True,
        #                         execution_engine=engine)

        engine = StreamingSequentialExecution(
            datasource=excerpts,
            policy=policy,
            nocache=True,
            verbose=False,
            allow_code_synth=False,
            allow_token_reduction=False,
            allow_bonded_query=True,
        )
        engine.generate_plan(output, policy)

        # physical_op_type = type('LLMBondedQueryConvert',
        #                 (LLMBondedQueryConvert,),
        #                 {'model': engine.plan.operators[1].model,
        #                  'prompt_strategy': pz.PromptStrategy.DSPY_COT_QA})
        #
        # bonded_convert = physical_op_type(
        #     input_schema=engine.plan.operators[1].input_schema,
        #     output_schema=engine.plan.operators[1].output_schema,
        #     query_strategy=pz.QueryStrategy.BONDED_WITH_FALLBACK,
        #     shouldProfile=False,
        #     cardinality=pz.Cardinality.ONE_TO_MANY,
        # )
        #
        # engine.plan.operators[1] = bonded_convert
        print("Generated plan:\n", engine.plan)
        with st.container():
            st.write("### Executed plan: \n")
            # st.write(" " + str(plan).replace("\n", "  \n "))
            for idx, op in enumerate(engine.plan.operators):
                strop = f"{idx + 1}. {str(op)}"
                strop = strop.replace("\n", "  \n")
                st.write(strop)

        input_records = engine.get_input_records()
        input_df = DataRecord.as_df(input_records, fields_in_schema=True)
        print(input_df)

        variables = []
        statistics = []
        start_time = time.time()
        # for idx, (vars, plan, stats) in enumerate(iterable):
        for idx, record in enumerate(input_records):
            print(f"idx: {idx}\n vars: {vars}")
            index = idx
            vars = engine.execute_opstream(engine.plan, record)
            if idx == len(input_records) - 1:
                total_plan_time = time.time() - start_time
                engine.plan_stats.finalize(total_plan_time)

            statistics.append(engine.plan_stats)
            intermediate_vars = DataRecord.as_df(vars, fields_in_schema=True)
            print(intermediate_vars)
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
        # with open(f"askem-variables-{dataset}.json", "w") as f:
        #     json.dump(variables, f, indent=4)
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
