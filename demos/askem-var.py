#!/usr/bin/env python3
"""This scripts is a demo for the biofabric data integration.
python src/cli/cli_main.py reg --path testdata/bdf-usecase3-pdf/ --name bdf-usecase3-pdf

"""

import json
import time

import pandas as pd
import streamlit as st

import palimpzest as pz


class Papersnippet(pz.TextFile):
    """Represents an excerpt from a scientific research paper, which potentially contains variables"""

    excerptid = pz.Field(desc="The unique identifier for the excerpt", required=True)
    excerpt = pz.Field(desc="The text of the excerpt", required=True)


class Variable(pz.Schema):
    """Represents a variable of scientific model in a scientific paper"""

    name = pz.Field(desc="The label used for a the scientific variable, like a, b, 𝜆 or 𝜖, NOT None", required=True)
    description = pz.Field(desc="A description of the variable, optional, set 'null' if not found", required=False)
    value = pz.Field(desc="The value of the variable, optional, set 'null' if not found", required=False)


if __name__ == "__main__":
    run_pz = True
    dataset = "askem"

    if run_pz:
        # reference, plan, stats = run_workload()
        excerpts = pz.Dataset(dataset, schema=pz.TextFile)
        output = excerpts.convert(
            Variable, desc="A variable used or introduced in the paper snippet", cardinality=pz.Cardinality.ONE_TO_MANY
        )

        engine = pz.StreamingSequentialExecution
        # policy = pz.MinCost()
        policy = pz.MaxQuality()
        # iterable  =  pz.Execute(output,
        #                         policy = policy,
        #                         nocache=True,
        #                         verbose=True,
        #                         allow_code_synth=False,
        #                         allow_token_reduction=False,
        #                         allow_bonded_query=True,
        #                         execution_engine=engine)

        engine = pz.StreamingSequentialExecution(
            policy=policy,
            nocache=True,
            verbose=True,
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
        print(engine.plan)
        with st.container():
            st.write("### Executed plan: \n")
            # st.write(" " + str(plan).replace("\n", "  \n "))
            for idx, op in enumerate(engine.plan.operators):
                strop = f"{idx + 1}. {str(op)}"
                strop = strop.replace("\n", "  \n")
                st.write(strop)

        input_records = engine.get_input_records()

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

            record_time = time.time()
            statistics.append(engine.plan_stats)

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

    print("References:", vars_df)
    # st.write(table.title, table.author, table.abstract)
    # endTime = time.time()
    # print("Elapsed time:", endTime - startTime)
