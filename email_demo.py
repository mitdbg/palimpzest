import argparse
import palimpzest as pz

import json
import os
import pandas as pd
from smolagents import tool, CodeAgent, LiteLLMModel

# Define tools for the agent to use
@tool
def list_filepaths() -> list[str]:
    """
    This tool lists all of the file paths for relevant files in the data directory.

    Args:
    None

    Returns:
    list[str]: A list of file paths for all files in the dataset directory.
    """
    dataset_directory = "testdata/enron-eval-medium"

    filepaths = []
    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if file.startswith("."):
                continue
            filepaths.append(os.path.join(root, file))
    return filepaths

@tool
def read_file(filepath: str) -> str:
    """
    This tool takes a file path as input and returns the content of the file as a string.
    It handles both CSV files and html / regular text files.

    Args:
    filepath: The path to the file to read.

    Returns:
    str: The content of the file as a string.
    """
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, encoding="ISO-8859-1").to_string(index=False)

    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    return content



def run_agents(model_id="anthropic/claude-3-7-sonnet-latest"):
    # ask the agent the question
    question = "Compute the sender and subject of every email which refers to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, and is not quoting articles or other sources outside of Enron"
    agent = CodeAgent(
        tools=[list_filepaths, read_file],
        model=LiteLLMModel(
            model_id=model_id,
            api_key=os.getenv("TIM_ANTHROPIC_API_KEY"),
        ),
        max_steps=20,
        planning_interval=4,
        add_base_tools=False,
        return_full_result=True,
    )
    result = agent.run(question)
    response = result.output
    input_tokens = result.token_usage.input_tokens
    output_tokens = result.token_usage.output_tokens
    cost_per_input_token = (3.0 / 1e6) if "anthropic" in model_id else (0.15 / 1e6)
    cost_per_output_token = (15.0 / 1e6) if "anthropic" in model_id else (0.6 / 1e6)
    input_cost = input_tokens * cost_per_input_token
    output_cost = output_tokens * cost_per_output_token
    return response, input_tokens, output_tokens, input_cost, output_cost


def run_pz():
    ds = pz.Dataset("testdata/enron-eval-medium")
    ds = ds.search(
        "The list of emails which refer to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, that are not quoting articles or other sources outside of Enron"
    )
    ds = ds.compute(
        "The sender and subject of each email"
    )
    return plan.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email demo script")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the script in")
    args = parser.parse_args()

    # execute script
    if args.mode == "pz":
        output = run_pz()
        output.to_df().to_csv("pz-email-output.csv", index=False)
    else:
        response, input_tokens, output_tokens, input_cost, output_cost = run_agents()
        response_dict = {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
        }
        with open("agents-email-output.csv", "w") as f:
            json.dump(response_dict, f)
