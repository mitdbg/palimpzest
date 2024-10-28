from __future__ import annotations

import hashlib
import json
from typing import Callable

from palimpzest.constants import MAX_ID_CHARS, AggFunc, Cardinality
from palimpzest.corelib.schemas import ImageFile, Schema
from palimpzest.elements.filters import Filter
from palimpzest.elements.groupbysig import GroupBySig


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

    Every logical operator must declare the get_op_params() method, which returns
    a dictionary of parameters that are used to implement its physical operator.
    """

    def __init__(
        self,
        output_schema: type[Schema],
        input_schema: type[Schema] | None = None,
    ):
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.op_id: str | None = None

    def __str__(self) -> str:
        raise NotImplementedError("Abstract method")

    def __eq__(self, other) -> bool:
        raise NotImplementedError("Calling __eq__ on abstract method")

    def copy(self) -> LogicalOperator:
        raise NotImplementedError("Abstract method")

    def op_name(self) -> str:
        """Name of the physical operator."""
        return str(self.__class__.__name__)

    def get_op_params(self):
        """
        Returns a dictionary mapping logical operator parameters which may be used to
        implement a physical operator associated with this logical operation.

        This method is also used to obtain the op_id for the logical operator.
        """
        raise NotImplementedError("Calling get_op_params on abstract method")

    def get_op_id(self):
        """
        NOTE: We do not call this in the __init__() method as subclasses may set parameters
              returned by self.get_op_params() after they call to super().__init__().

        NOTE: This is NOT a universal ID.

        Two different PhysicalOperator instances with the identical returned values
        from the call to self.get_op_params() will have equivalent op_ids.
        """
        # return self.op_id if we've computed it before
        if self.op_id is not None:
            return self.op_id

        # compute, set, and return the op_id
        op_name = self.op_name()
        op_params = self.get_op_params()
        op_params = {k: str(v) for k, v in op_params.items()}
        hash_str = json.dumps({"op_name": op_name, **op_params}, sort_keys=True)
        self.op_id = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

        return self.op_id

    def __hash__(self):
        if not self.op_id:
            raise ValueError("op_id not set, unable to hash")
        return int(self.op_id, 16)


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

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, Aggregate)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.agg_func == other.agg_func
        )

    def copy(self):
        return self.__class__(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            agg_func=self.agg_func,
            target_cache_id=self.target_cache_id,
        )

    def get_op_params(self) -> dict:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "agg_func": self.agg_func,
            "target_cache_id": self.target_cache_id,
        }


class BaseScan(LogicalOperator):
    """A BaseScan is a logical operator that represents a scan of a particular data source."""

    def __init__(self, dataset_id: str, *args, **kwargs):
        if kwargs.get("input_schema") is not None:
            raise Exception(
                f"BaseScan must be initialized with `input_schema=None` but was initialized with "
                f"`input_schema={kwargs.get('input_schema')}`"
            )

        super().__init__(*args, **kwargs)
        self.dataset_id = dataset_id

    def __str__(self):
        return f"BaseScan({self.dataset_id},{str(self.output_schema)})"

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, BaseScan)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.dataset_id == other.dataset_id
        )

    def copy(self):
        return BaseScan(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_id=self.dataset_id,
        )

    def get_op_params(self) -> dict:
        return {"output_schema": self.output_schema, "dataset_id": self.dataset_id}


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

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, CacheScan)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.dataset_id == other.dataset_id
        )

    def copy(self):
        return CacheScan(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_id=self.dataset_id,
        )

    def get_op_params(self):
        return {
            "output_schema": self.output_schema,
            "dataset_id": self.dataset_id,
        }


class ConvertScan(LogicalOperator):
    """A ConvertScan is a logical operator that represents a scan of a particular data source,
    with conversion applied."""

    def __init__(
        self,
        cardinality: Cardinality = Cardinality.ONE_TO_ONE,
        udf: Callable | None = None,
        image_conversion: bool = False,
        depends_on: list[str] | None = None,
        desc: str | None = None,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cardinality = cardinality
        self.udf = udf
        self.image_conversion = image_conversion or (self.input_schema == ImageFile)
        self.depends_on = [] if depends_on is None else depends_on
        self.desc = desc
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"ConvertScan({self.input_schema} -> {str(self.output_schema)},{str(self.desc)})"

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, ConvertScan)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.cardinality == other.cardinality
            and self.udf == other.udf
            and self.image_conversion == other.image_conversion
            and self.depends_on == other.depends_on
        )

    def copy(self):
        return ConvertScan(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            cardinality=self.cardinality,
            image_conversion=self.image_conversion,
            udf=self.udf,
            depends_on=self.depends_on,
            desc=self.desc,
            target_cache_id=self.target_cache_id,
        )

    def get_op_params(self):
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cardinality": self.cardinality,
            "udf": self.udf,
            "image_conversion": self.image_conversion,
            "desc": self.desc,
            "target_cache_id": self.target_cache_id,
        }


class FilteredScan(LogicalOperator):
    """A FilteredScan is a logical operator that represents a scan of a particular data source, with filters applied."""

    def __init__(
        self,
        filter: Filter,
        image_filter: bool = False,
        depends_on: list[str] | None = None,
        target_cache_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.filter = filter
        self.image_filter = image_filter or (self.input_schema == ImageFile)
        self.depends_on = [] if depends_on is None else depends_on
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"FilteredScan({str(self.output_schema)}, {str(self.filter)})"

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, FilteredScan)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.filter == other.filter
            and self.image_filter == other.image_filter
        )

    def copy(self):
        return FilteredScan(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            filter=self.filter,
            image_filter=self.image_filter,
            depends_on=self.depends_on,
            target_cache_id=self.target_cache_id,
        )

    def get_op_params(self) -> dict:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "filter": self.filter,
            "image_filter": self.image_filter,
            "target_cache_id": self.target_cache_id,
        }


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

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, GroupByAggregate)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.group_by_sig == other.group_by_sig
        )

    def copy(self):
        return GroupByAggregate(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            group_by_sig=self.group_by_sig,
            target_cache_id=self.target_cache_id,
        )

    def get_op_params(self) -> dict:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "group_by_sig": self.group_by_sig,
            "target_cache_id": self.target_cache_id,
        }


class LimitScan(LogicalOperator):
    def __init__(self, limit: int, target_cache_id: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.target_cache_id = target_cache_id

    def __str__(self):
        return f"LimitScan({str(self.input_schema)}, {str(self.output_schema)})"

    def copy(self):
        return LimitScan(
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            limit=self.limit,
            target_cache_id=self.target_cache_id,
        )

    def __eq__(self, other: LogicalOperator) -> bool:
        return (
            isinstance(other, LimitScan)
            and self.input_schema == other.input_schema
            and self.output_schema == other.output_schema
            and self.limit == other.limit
        )

    def get_op_params(self) -> dict:
        return {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "limit": self.limit,
            "target_cache_id": self.target_cache_id,
        }
