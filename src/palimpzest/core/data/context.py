from __future__ import annotations

import os
import re
from abc import ABC
from typing import Callable

from pydantic import BaseModel
from smolagents import CodeAgent, LiteLLMModel

from palimpzest.core.data import context_manager
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import create_schema_from_fields, union_schemas
from palimpzest.query.operators.logical import ComputeOperator, ContextScan, LogicalOperator, SearchOperator
from palimpzest.utils.hash_helpers import hash_for_id

PZ_INSTRUCTION = """\n\nYou are a CodeAgent who is a specialist at writing declarative AI programs with the Palimpzest (PZ) library.

Palimpzest is a programming framework which provides you with **semantic operators** (e.g. semantic maps, semantic filters, etc.)
which are like their traditional counterparts, except they can execute instructions provided in natural language.

For example, if you wanted to write a program to extract the title and abstract from a directory of papers,
you could write the following in PZ:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define columns for semantic map (sem_map) operation; each column is specified
# with a dictionary containing the following keys:
# - "name": the name of the field to compute
# - "type": the type of the field to compute
# - "description": the natural language description of the field
paper_cols = [
    {"name": "title", "type": str, "description": "the title of the paper"},
    {"name": "abstract", "type": str, "description": "the paper's abstract"},
]

# construct the data processing pipeline with PZ
ds = pz.TextFileDataset(id="papers", path="path/to/papers")
ds = ds.sem_map(cols)

# optimize and execute the PZ program
validator = pz.Validator()
config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    execution_strategy="parallel",
    max_workers=20,
    progress=True,
)
output = ds.optimize_and_run(config=config, validator=validator)

# write the execution stats to json
output.execution_stats.to_json("pz_program_stats.json")

# write the output to a CSV and print the output CSV filepath so the user knows where to find it
output_filepath = "pz_program_output.csv"
output.to_df().to_csv(output_filepath, index=False)
print(f"Results at: {output_filepath}")
```

To initialize a dataset in PZ, simply provide the path to a directory to `pz.TextFileDirectory()`
(if your data contains text-based files). For example:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ds = pz.TextFileDataset(id="files", path="path/to/files")
```

Palimpzest has two primary **semantic operators** which you can use to construct data processing pipelines:
- sem_filter(predicate: str): executes a semantic filter specified by the natural language predicate on a given PZ dataset
- sem_map(cols: list[dict]): executes a semantic map to compute the `cols` on a given PZ dataset

As a second example, consider the following PZ program which filters for papers about batteries that are from MIT
and computes a summary for each one:
```
import palimpzest as pz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# construct the PZ program
ds = pz.TextFileDataset(id="papers", path="path/to/research-papers")
ds = ds.sem_filter("The paper is about batteries")
ds = ds.sem_filter("The paper is from MIT")
ds = ds.sem_map([{"name": "summary", "type": str, "description": "A summary of the paper"}])

# optimize and execute the PZ program
validator = pz.Validator()
config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    execution_strategy="parallel",
    max_workers=20,
    progress=True,
)
output = ds.optimize_and_run(config=config, validator=validator)

# write the execution stats to json
output.execution_stats.to_json("pz_program_stats.json")

# write the output to a CSV and print the output CSV filepath so the user knows where to find it
output_filepath = "pz_program_output.csv"
output.to_df().to_csv(output_filepath, index=False)
print(f"Results at: {output_filepath}")
```

Be sure to always:
- execute your program using the `.optimize_and_run()` format shown above
- call `output.execution_stats.to_json("pz_program_stats.json")` to write execution statistics to disk
- write your output to CSV and print where you wrote it!
"""

class Context(Dataset, ABC):
    """
    The `Context` class is an abstract base class for root `Datasets` whose data is accessed
    via user-defined methods. Classes which inherit from this class must implement two methods:

    - `list_filepaths()`: which lists the files that the `Context` has access to.
    - `read_filepath(path: str)`: which reads the file corresponding to the given `path`.

    A `Context` is a special type of `Dataset` that represents a view over an underlying `Dataset`.
    Each `Context` has a `name` which uniquely identifies it, as well as a natural language `description`
    of the data / computation that the `Context` represents. Similar to `Dataset`s, `Context`s can be
    lazily transformed using functions such as `sem_filter`, `sem_map`, `sem_join`, etc., and they may
    be materialized or unmaterialized.
    """

    def __init__(
            self,
            id: str,
            description: str,
            operator: LogicalOperator,
            schema: type[BaseModel] | None = None,
            sources: list[Context] | Context | None = None,
            materialized: bool = False,
        ) -> None:
        """
        Constructor for the `Context` class.

        Args:
            id (`str`): a string identifier for the `Context`
            description (`str`): the description of the data contained within the `Context`
            operator (`LogicalOperator`): The `LogicalOperator` used to compute this `Context`.
            schema: (`type[BaseModel] | None`): The schema of this `Context`.
            sources (`list[Context] | Context | None`): The (list of) `Context(s)` which are input(s) to
                the operator used to compute this `Context`.
            materialized (`bool`): True if the `Context` has been computed, False otherwise
        """
        # set the description
        self._description = description

        # set the materialization status
        self._materialized = materialized

        # compute schema and call parent constructor
        if schema is None:
            schema = create_schema_from_fields([{"name": "context", "description": "The context", "type": str}])
        super().__init__(sources=sources, operator=operator, schema=schema, id=id)

        # set the tools associated with this Context
        self._tools = [getattr(self, attr) for attr in dir(self) if attr.startswith("tool_")]

        # add Context to ContextManager
        cm = context_manager.ContextManager()
        cm.add_context(self)

    @property
    def description(self) -> str:
        """The string containing all of the information computed for this `Context`"""
        return self._description

    @property
    def materialized(self) -> bool:
        """The boolean which specifies whether the `Context` has been computed or not"""
        return self._materialized

    @property
    def tools(self) -> list[Callable]:
        """The list of tools associated with this `Context`"""
        return self._tools

    def __str__(self) -> str:
        return f"Context(id={self.id}, description={self.description:20s}, materialized={self.materialized})"

    def set_description(self, description: str) -> None:
        """
        Update the context's description.
        """
        self._description = description

    def set_materialized(self, materialized: str) -> None:
        """
        Update the context's materialization status.
        """
        self._materialized = materialized

    def compute(self, instruction: str) -> Context:
        # construct new description and output schema
        new_id = hash_for_id(instruction)
        new_description = f"Parent Context ID: {self.id}\n\nThis Context is the result of computing the following instruction on the parent context.\n\nINSTRUCTION: {instruction}\n\n"
        inter_schema = create_schema_from_fields([{"name": f"result-{new_id}", "desc": "The result from computing the instruction on the input Context",  "type": str}])
        new_output_schema = union_schemas([self.schema, inter_schema])

        # construct logical operator
        operator = ComputeOperator(
            input_schema=self.schema,
            output_schema=new_output_schema,
            context_id=new_id,
            instruction=instruction,
        )        

        return Context(id=new_id, description=new_description, operator=operator, sources=[self], materialized=False)

    def search(self, search_query: str) -> Context:
        # construct new description and output schema
        new_id = hash_for_id(search_query)
        new_description = f"Parent Context ID: {self.id}\n\nThis Context is the result of searching the parent context for information related to the following query.\n\nSEARCH QUERY: {search_query}\n\n"

        # construct logical operator
        operator = SearchOperator(
            input_schema=self.schema,
            output_schema=self.schema,
            context_id=new_id,
            search_query=search_query,
        )

        return Context(id=new_id, description=new_description, operator=operator, sources=[self], materialized=False)

class TextFileContext(Context):
    def __init__(self, path: str, id: str, description: str) -> None:
        """
        Constructor for the `TextFileContext` class.

        Args:
            path (str): The path to the file
            id (str): a string identifier for the `Context`
            description (str): The description of the data contained within the `Context`
            kwargs (dict): Keyword arguments containing the `Context's` id and description.
        """
        # check that path is a valid file or directory
        assert os.path.isfile(path) or os.path.isdir(path), f"Path {path} is not a file nor a directory"

        # get list of filepaths
        self.filepaths = []
        if os.path.isfile(path):
            self.filepaths = [path]
        else:
            self.filepaths = []
            for root, _, files in os.walk(path):
                for file in files:
                    fp = os.path.join(root, file)
                    self.filepaths.append(fp)
            self.filepaths = sorted(self.filepaths)

        # call parent constructor to set id, operator, and schema
        schema = create_schema_from_fields([{"name": "context", "desc": "The context", "type": str}])
        super().__init__(
            id=id,
            description=description,
            operator=ContextScan(context=self, output_schema=schema),
            schema=schema,
            materialized=True,
        )
    def _check_filter_answer_text(self, answer_text: str) -> dict | None:
        """
        Return {"passed_operator": True} if and only if "true" is in the answer text.
        Return {"passed_operator": False} if and only if "false" is in the answer text.
        Otherwise, return None.
        """
        # NOTE: we may be able to eliminate this condition by specifying this JSON output in the prompt;
        # however, that would also need to coincide with a change to allow the parse_answer_fn to set "passed_operator"
        if "true" in answer_text.lower():
            return {"passed_operator": True}
        elif "false" in answer_text.lower():
            return {"passed_operator": False}
        elif "yes" in answer_text.lower():
            return {"passed_operator": True}

        return None

    def _parse_filter_answer(self, completion_text: str) -> dict[str, list]:
        """Extract the answer from the completion object for filter operations."""
        # if the model followed the default instructions, the completion text will place
        # its answer between "ANSWER:" and "---"
        regex = re.compile("answer:(.*?)---", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # if the first regex didn't find an answer, try taking all the text after "ANSWER:"
        regex = re.compile("answer:(.*)", re.IGNORECASE | re.DOTALL)
        matches = regex.findall(completion_text)
        if len(matches) > 0:
            answer_text = matches[0].strip()
            field_answers = self._check_filter_answer_text(answer_text)
            if field_answers is not None:
                return field_answers

        # finally, try taking all of the text; throw an exception if this doesn't work
        field_answers = self._check_filter_answer_text(completion_text)
        if field_answers is None:
            raise Exception(f"Could not parse answer from completion text: {completion_text}")

        return field_answers

    # def tool_list_filepaths(self) -> list[str]:
    #     """
    #     This tool returns the list of all of the filepaths which the `Context` has access to.

    #     Args:
    #         None
        
    #     Returns:
    #         list[str]: A list of file paths for all files in the `Context`.
    #     """
    #     return self.filepaths

    # def tool_read_filepath(self, path: str) -> str:
    #     """
    #     This tool takes a filepath (`path`) as input and returns the content of the file as a string.
    #     It handles both CSV files and html / regular text files. It does not handle images.

    #     Args:
    #         path (str): The path to the file to read.

    #     Returns:
    #         str: The content of the file as a string.
    #     """
    #     if path.endswith(".csv"):
    #         return pd.read_csv(path, encoding="ISO-8859-1").to_string(index=False)

    #     with open(path, encoding='utf-8') as file:
    #         content = file.read()

    #     return content

    def tool_execute_semantic_operators(self, instruction: str) -> str:
        """
        This tool takes an `instruction` as input and invokes an expert to write a semantic data processing pipeline
        to execute the instruction. The tool returns the path to a CSV file which contains the output of the pipeline.

        For example, the tool could be invoked as follows to extract the title and abstract from a dataset of research papers:
        ```
        instruction = "Write a program to extract the title and abstract from each research paper"
        result_csv_filepath = tool_execute_semantic_operators(instruction)
        ```

        Args:
            instruction: The instruction specifying the semantic data processing pipeline that you need to execute.

        Returns:
            str: the filepath to the CSV containing the output from running the data processing pipeline.
        """
        from smolagents import tool
        @tool
        def tool_list_filepaths() -> list[str]:
            """
            This tool returns the list of all of the filepaths which the `Context` has access to.

            NOTE: You may want to execute this before writing your PZ program to determine where the data lives.

            Args:
                None
            
            Returns:
                list[str]: A list of file paths for all files in the `Context`.
            """
            return self.filepaths

        agent = CodeAgent(
            model=LiteLLMModel(model_id="openai/o1", api_key=os.getenv("ANTHROPIC_API_KEY")),
            tools=[tool_list_filepaths],
            max_steps=20,
            planning_interval=4,
            add_base_tools=False,
            return_full_result=True,
            additional_authorized_imports=["dotenv", "json", "palimpzest", "pandas"],
            instructions=PZ_INSTRUCTION,
        )
        result = agent.run(instruction)
        response = result.output

        return response
