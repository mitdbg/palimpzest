from __future__ import annotations

import os
import re
from abc import ABC
from typing import Callable

import pandas as pd
from pydantic import BaseModel

from palimpzest.core.data import context_manager
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import create_schema_from_fields, union_schemas
from palimpzest.query.operators.logical import ComputeOperator, ContextScan, LogicalOperator, SearchOperator
from palimpzest.utils.hash_helpers import hash_for_id


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

    def tool_list_filepaths(self) -> list[str]:
        """
        This tool returns the list of all of the filepaths which the `Context` has access to.

        Args:
            None
        
        Returns:
            list[str]: A list of file paths for all files in the `Context`.
        """
        return self.filepaths

    def tool_read_filepath(self, path: str) -> str:
        """
        This tool takes a filepath (`path`) as input and returns the content of the file as a string.
        It handles both CSV files and html / regular text files. It does not handle images.

        Args:
            path (str): The path to the file to read.

        Returns:
            str: The content of the file as a string.
        """
        if path.endswith(".csv"):
            return pd.read_csv(path, encoding="ISO-8859-1").to_string(index=False)

        with open(path, encoding='utf-8') as file:
            content = file.read()

        return content
