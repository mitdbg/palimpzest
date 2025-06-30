from __future__ import annotations

import os
from abc import ABC
from typing import Callable

import pandas as pd

from palimpzest.core.data import context_manager
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.lib.schemas import Schema
from palimpzest.query.operators.logical import ComputeOperator, ContextScan, LogicalOperator
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
            schema: type[Schema] | None = None,
            sources: list[Context] | Context | None = None,
            materialized: bool = False,
        ) -> None:
        """
        Constructor for the `Context` class.

        Args:
            id (`str`): a string identifier for the `Context`
            description (`str`): the description of the data contained within the `Context`
            operator (`LogicalOperator`): The `LogicalOperator` used to compute this `Context`.
            schema: (`type[Schema] | None`): 
            sources (`list[Context] | Context | None`): The (list of) `Context(s)` which are input(s) to
                the operator used to compute this `Context`.
            materialized (`bool`): True if the `Context` has been computed, False otherwise
        """
        # set the description
        self._description = description

        # set the materialization status
        self._materialized = materialized

        # compute Schema and call parent constructor
        if schema is None:
            schema = Schema.from_fields([{"name": "context", "desc": "The context", "type": str}])
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
        new_output_schema = self.schema.add_fields([
            {"name": f"instruction-{new_id}", "desc": "The instruction used to compute this Context", "type": str},
            {"name": f"result-{new_id}", "desc": "The result from computing the instruction on the input Context",  "type": str}
        ])

        # construct logical operator
        operator = ComputeOperator(
            input_schema=self.schema,
            output_schema=new_output_schema,
            context_id=new_id,
            instruction=instruction,
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
        schema = Schema.from_fields([{"name": "context", "desc": "The context", "type": str}])
        super().__init__(
            id=id,
            description=description,
            operator=ContextScan(context=self, output_schema=schema),
            schema=schema,
            materialized=True,
        )

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
        It handles both CSV files and html / regular text files.

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
