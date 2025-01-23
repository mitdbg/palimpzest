from __future__ import annotations

import json
from typing import Callable

from palimpzest.constants import AggFunc, Cardinality
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.lib.schemas import Schema
from palimpzest.utils.hash_helpers import hash_for_id


class LogicalOperator:
    """
    A logical operator is an operator that operates on Sets.

    Right now it can be one of:
    - BaseScan (scans data from DataSource)
    - CacheScan (scans cached Set)
    - FilteredScan (scans input Set and applies filter)
    - ConvertScan (scans input Set and converts it to new Schema)
    - LimitScan (scans up to N records from a Set)
    - GroupByAggregate (applies a group by on the Set)
    - Aggregate (applies an aggregation on the Set)
    - RetrieveScan (fetches documents from a provided input for a given query)

    Every logical operator must declare the get_logical_id_params() and get_logical_op_params() methods,
    which return dictionaries of parameters that are used to compute the logical op id and to implement
    the logical operator (respectively).
    """

    def __init__(
        self,
        output_schema: Schema,
        input_schema: Schema | None = None,
    ):
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.logical_op_id: str | None = None

    def __str__(self) -> str:
        raise NotImplementedError("Abstract method")

    def __eq__(self, other) -> bool:
        all_id_params_match = all(value == getattr(other, key) for key, value in self.get_logical_id_params().items())
        return isinstance(other, self.__class__) and all_id_params_match

    def copy(self) -> LogicalOperator:
        return self.__class__(**self.get_logical_op_params())

    def logical_op_name(self) -> str:
        """Name of the logical operator."""
        return str(self.__class__.__name__)

    def get_logical_id_params(self) -> dict:
        """
        Returns a dictionary mapping of logical operator parameters which are relevant
        for computing the logical operator id.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        NOTE: input_schema is not included in the id params because it depends on how the Optimizer orders operations.
        """
        return {"output_schema": self.output_schema}

    def get_logical_op_params(self) -> dict:
        """
        Returns a dictionary mapping of logical operator parameters which may be used to
        implement a physical operator associated with this logical operation.
        
        NOTE: Should be overriden by subclasses to include class-specific parameters.
        """
        return {"input_schema": self.input_schema, "output_schema": self.output_schema}

    def get_logical_op_id(self):
        """
        NOTE: We do not call this in the __init__() method as subclasses may set parameters
              returned by self.get_logical_op_params() after they call to super().__init__().
        """
        # return self.logical_op_id if we've computed it before
        if self.logical_op_id is not None:
            return self.logical_op_id

        # get op name and op parameters which are relevant for computing the id
        logical_op_name = self.logical_op_name()
        logical_id_params = self.get_logical_id_params()
        logical_id_params = {k: str(v) for k, v in logical_id_params.items()}

        # compute, set, and return the op_id
        hash_str = json.dumps({"logical_op_name": logical_op_name, **logical_id_params}, sort_keys=True)
        self.logical_op_id = hash_for_id(hash_str)

        return self.logical_op_id

    def __hash__(self):
        if not self.logical_op_id:
            raise ValueError("logical_op_id not set, unable to hash")
        return int(self.logical_op_id, 16)


class Aggregate(LogicalOperator):
    """
    Aggregate is a logical operator that applies an aggregation to the input set and yields a single result.
    This is a base class that has to be further specialized to implement specific aggregation functions.
    """

    def __init__(
        self,
        agg_func: AggFunc,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"{self.__class__.__name__}(function: {str(self.agg_func.value)})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"agg_func": self.agg_func, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "agg_func": self.agg_func,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""

    def __init__(self, dataset_id: str, output_schema: Schema):
        super().__init__(output_schema=output_schema)
        self.dataset_id = dataset_id

    def __str__(self):
        return f"BaseScan({self.dataset_id},{str(self.output_schema)})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, BaseScan)
            and self.input_schema.get_desc() == other.input_schema.get_desc()
            and self.output_schema.get_desc() == other.output_schema.get_desc()
            and self.dataset_id == other.dataset_id
        )

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"dataset_id": self.dataset_id, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"dataset_id": self.dataset_id, **logical_op_params}

        return logical_op_params


class CacheScan(LogicalOperator):
    """A CacheScan is a logical operator that represents a scan of a cached Set."""

    def __init__(self, dataset_id: str, *args, **kwargs):
        if kwargs.get("input_schema") is not None:
            raise Exception(
                f"CacheScan must be initialized with `input_schema=None` but was initialized with "
                f"`input_schema={kwargs.get('input_schema')}`"
            )

        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id

    def __str__(self):
        return f"CacheScan({str(self.output_schema)},{str(self.dataset_id)})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"dataset_id": self.dataset_id, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {"dataset_id": self.dataset_id, **logical_op_params}

        return logical_op_params


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source, with conversion applied."""

    def __init__(
        self,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        udf: Callable | None = None,
        depends_on: list[str] | None = None,
        desc: str | None = None,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cardinality = cardinality
        self.udf = udf
        self.depends_on = [] if depends_on is None else sorted(depends_on)
        self.desc = desc
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"ConvertScan({self.input_schema} -> {str(self.output_schema)},{str(self.desc)})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {
            "cardinality": self.cardinality,
            "udf": self.udf,
            **logical_id_params,
        }

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "cardinality": self.cardinality,
            "udf": self.udf,
            "depends_on": self.depends_on,
            "desc": self.desc,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""

    def __init__(
        self,
        filter: Filter,
        depends_on: list[str] | None = None,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.filter = filter
        self.depends_on = [] if depends_on is None else sorted(depends_on)
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"FilteredScan({str(self.output_schema)}, {str(self.filter)})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {
            "filter": self.filter,
            **logical_id_params,
        }

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "filter": self.filter,
            "depends_on": self.depends_on,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params

class GroupByAggregate(LogicalOperator):
    def __init__(
        self,
        group_by_sig: GroupBySig,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.input_schema:
            raise ValueError("GroupByAggregate requires an input schema")
        (valid, error) = group_by_sig.validate_schema(self.input_schema)
        if not valid:
            raise TypeError(error)
        self.group_by_sig = group_by_sig
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"GroupBy({self.group_by_sig.serialize()})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"group_by_sig": self.group_by_sig, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "group_by_sig": self.group_by_sig,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params

class LimitScan(LogicalOperator):
    def __init__(self, limit: int, target_cache_id: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"LimitScan({str(self.input_schema)}, {str(self.output_schema)})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"limit": self.limit, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "limit": self.limit,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params


class Project(LogicalOperator):
    def __init__(self, project_cols: list[str], target_cache_id: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_cols = project_cols
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"Project({self.input_schema}, {self.project_cols})"

    def get_logical_id_params(self) -> dict:
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {"project_cols": self.project_cols, **logical_id_params}

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "project_cols": self.project_cols,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params


class RetrieveScan(LogicalOperator):
    """A RetrieveScan is a logical operator that represents a scan of a particular data source, with a convert-like retrieve applied."""

    def __init__(
        self,
        index,
        search_attr,
        output_attr,
        k,
        target_cache_id: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index = index
        self.search_attr = search_attr
        self.output_attr = output_attr  
        self.k = k
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"RetrieveScan({self.input_schema} -> {str(self.output_schema)},{str(self.desc)})"

    def get_logical_id_params(self) -> dict:
        # NOTE: if we allow optimization over index, then we will need to include it in the id params
        # NOTE: ^if we do this, we should probably make a wrapper around the index object to ensure that
        #       it can be serialized as a string properly
        logical_id_params = super().get_logical_id_params()
        logical_id_params = {
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            **logical_id_params,
        }

        return logical_id_params

    def get_logical_op_params(self) -> dict:
        logical_op_params = super().get_logical_op_params()
        logical_op_params = {
            "index": self.index,
            "search_attr": self.search_attr,
            "output_attr": self.output_attr,
            "k": self.k,
            "target_cache_id": self.target_cache_id,
            **logical_op_params,
        }

        return logical_op_params
