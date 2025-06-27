from __future__ import annotations

import os
import pickle
from abc import ABC

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
        ) -> None:
        """
        Constructor for the `Context` class.

        Args:
            id (str): a string identifier for the `Context`
            description (str): the description of the data contained within the `Context`
            operator (`LogicalOperator`): The `LogicalOperator` used to compute this `Context`.
            schema: (`type[Schema]` | None): 
            sources (`list[Context] | Context | None`): The (list of) `Context(s)` which are input(s) to
                the operator used to compute this `Context`.
        """
        # set the id for the Dataset
        self._id = id

        # set the description
        self._description = description

        # set the logical operator
        self._operator: LogicalOperator = operator

        # initialize an empty string for the context
        self._context = ""

        # compute Schema and call parent constructor
        if schema is None:
            schema = Schema.from_fields([{"name": f"context-{id}", "desc": "The context", "type": str}])
        super().__init__(sources=sources, operator=operator, schema=schema)

        # add Context to ContextManager
        cm = context_manager.ContextManager()
        cm.add_context(self)

    @property
    def context(self) -> str:
        """The string containing all of the information computed for this `Context`"""
        return self._context

    def __str__(self) -> str:
        return f"Context(id={self.id}, description={self._description:20s})"

    # TODO: maybe only need to pickle Dataset? Can we remove the to_json and from_json methods?
    @classmethod
    def from_pkl(cls, path: str) -> Context:
        """Load a `Context` from its serialized pickle file."""
        with open(path, "rb") as f:
            context = pickle.load(f)

        return context

        # # parse JSON for serialized context
        # id = context_json["id"]
        # description = context_json["description"]
        # op_class_name = context_json["op_class_name"]
        # op_kwargs = context_json["op_kwargs"]
        # schema_json = context_json["schema"]
        # sources_json = context_json["sources"]

        # # reconstruct logical operator
        # operator = None
        # for name, op_cls in inspect.getmembers(logical):
        #     if name == op_class_name:
        #         operator = op_cls(**op_kwargs)
        #         break

        # # assert that operator is found
        # assert operator is not None, f"Could not find operator class for operator with class name: {op_class_name}"

        # # reconstruct sources
        # sources = None
        # if sources_json is not None:
        #     sources = []
        #     for source_json in sources_json:
        #         source = Context.from_json(source_json) if source_json["type"] == "Context" else Dataset.from_json(source_json)
        #         sources.append(source)

        # return cls(id, description, operator, Schema.from_json(schema_json), sources)

    def to_pkl(self, path: str) -> None:
        """Write this `Context` to a pickle file at the provided `path`."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # return {
        #     "type": "Context",
        #     "id": self.id,
        #     "description": self._description,
        #     "op_class_name": self._operator.__class__.__name__,
        #     "op_kwargs": self._operator.get_logical_op_params(),
        #     "schema": self.schema.to_json(),
        #     "sources": None if self._sources is None else [source.to_json() for source in self._sources]
        # }

    def compute(self, instruction: str) -> Context:
        # construct new description and output schema
        new_id = hash_for_id(instruction)
        new_description = f"This Context is the result of computing the following instruction: {instruction}"
        new_output_schema = self.schema.add_fields([
            {"name": f"instruction-{new_id}", "desc": "The instruction used to compute this Context", "type": str},
            {"name": f"result-{new_id}", "desc": "The result from computing the instruction on the input Context",  "type": str}
        ])

        # construct logical operator
        operator = ComputeOperator(
            input_schema=self.schema,
            output_schema=new_output_schema,
            instruction=instruction,
        )        

        return Context(id=new_id, description=new_description, operator=operator, sources=[self])


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
        super().__init__(id=id, description=description, operator=ContextScan(context=self, output_schema=schema), schema=schema)

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
