import argparse
import json
import os
import time

import pandas as pd

# from colorama import Fore, Style
from pydantic import BaseModel, Field
from smolagents import CodeAgent, LiteLLMModel, tool

import palimpzest as pz
from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile, create_schema_from_fields
from palimpzest.query.generators.generators import Generator


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

    with open(filepath, encoding='utf-8') as file:
        content = file.read()

    return content


@tool
def sem_filter(filepath: str, predicate: str) -> bool:
    """
    This tool takes a file path and a natural language predicate as input and returns True
    if the file satisfies the natural language predicate and False otherwise.

    For example, the tool could be invoked as follows to see if a paper is about batteries:
    ```
    out = sem_filter("path/to/paper.txt", "This paper is about batteries")
    ```

    Args:
        filepath: The path to the file to read.
        predicate: A natural language predicate (i.e. a filter condition)

    Returns:
        bool: True if the file satisfies the predicate and False otherwise
    """
    with open(filepath) as f:
        contents = f.read()

    # create data record for file
    dr = DataRecord(schema=TextFile, source_indices=[f"file-{0}"])
    dr.filename = os.path.basename(filepath)
    dr.contents = contents

    # create output schema
    class Answer(BaseModel):
        passed_operator: bool = Field(description="Whether the record passed the filter operation")

    # call generator
    generator = Generator(Model.GPT_4o, PromptStrategy.COT_BOOL)
    gen_kwargs = {"filter_condition": predicate}
    output, _, _, _ = generator(dr, Answer.model_fields, **gen_kwargs)
    # output, _, _, _, prompt, completion_text = generator(dr, Answer.model_fields, **gen_kwargs)
    # print(f"PROMPT:\n{prompt}")
    # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

    return output["passed_operator"]


@tool
def sem_map(filepath: str, fields: list[dict]) -> dict:
    """
    This tool takes a file path and a list of dictionaries representing fields to compute as input
    and returns the dictionary mapping each field to its computed value. The input dictionary should
    contain key-value pairs for the `name`, `type`, and `description` of each field.

    For example, the tool could be invoked as follows to extract the title and abstract of a paper:
    ```
    fields = [
        {"name": "title", "type": str, "description": "the title of the paper"},
        {"name": "abstract", "type": str, "description": "the paper's abstract"},
    ]
    out = sem_map("path/to/paper.txt", fields)
    print(f"Title: {out['title']}")
    print(f"Abstract: {out['abstract']}")
    ```

    Args:
        filepath: The path to the file to read.
        fields: Dictionary of fields to compute. Keys are field names and values are field descriptions.

    Returns:
        dict: Dictionary mapping each field to its computed value
    """
    with open(filepath) as f:
        contents = f.read()

    # create data record for file
    dr = DataRecord(schema=TextFile, source_indices=[f"file-{0}"])
    dr.filename = os.path.basename(filepath)
    dr.contents = contents

    # create output schema
    output_schema = create_schema_from_fields(fields)

    # call generator
    generator = Generator(Model.GPT_4o, PromptStrategy.COT_QA)
    gen_kwargs = {"output_schema": output_schema}
    output, _, _, _ = generator(dr, output_schema.model_fields, **gen_kwargs)
    # output, _, _, _, _, _ = generator(dr, output_schema.model_fields, **gen_kwargs)
    output = {k: v[0] for k, v in output.items()}

    return output


def run_agents(model_id="anthropic/claude-3-5-sonnet-latest", api_key=None):
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    # ask the agent the question
    question = "Compute the sender, subject, and summary of every email which refers to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, and is not quoting articles or other sources outside of Enron"
    # question = "Compute the sender and subject of every email which refers to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, and is not quoting articles or other sources outside of Enron"
    agent = CodeAgent(
        tools=[list_filepaths, read_file, sem_filter, sem_map],
        model=LiteLLMModel(model_id=model_id, api_key=api_key),
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
    ds = pz.TextFileContext("testdata/enron-eval-medium", "enron-data", "250 emails from Enron employees")
    ds = ds.compute(
        "Compute the sender, subject, and summary of every email which refers to the Raptor, Deathstar, Chewco, and/or Fat Boy investments, and is not quoting text from an article or source outside of Enron"
    )
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        progress=False,
    )
    return ds.run(config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Email demo script")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the script in")
    args = parser.parse_args()

    start_time = time.time()

    # execute script
    if args.mode == "pz":
        output = run_pz()
        output.to_df().to_csv("pz-email-output.csv", index=False)

    else:
        response, input_tokens, output_tokens, input_cost, output_cost = run_agents(model_id="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
        response_dict = {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
        }
        with open("agents-email-output.json", "w") as f:
            json.dump(response_dict, f)

    print(f"TOTAL TIME: {time.time() - start_time:.2f}")
