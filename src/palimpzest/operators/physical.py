from __future__ import annotations

import hashlib
import json
from typing import List, Tuple

from palimpzest.constants import MAX_ID_CHARS
from palimpzest.corelib.schemas import Schema
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements.records import DataRecord

# TYPE DEFINITIONS
DataRecordsWithStats = Tuple[List[DataRecord], List[RecordOpStats]]


class PhysicalOperator:
    """
    All implemented physical operators should inherit from this class.
    In order for the Optimizer to consider using a physical operator for a
    given logical operation, the user must also write an ImplementationRule.
    """

    def __init__(
        self,
        output_schema: type[Schema],
        input_schema: type[Schema] | None = None,
        logical_op_id: str | None = None,
        max_workers: int = 1,
        target_cache_id: str | None = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.datadir = DataDirectory()
        self.max_workers = max_workers
        self.target_cache_id = target_cache_id
        self.verbose = verbose
        self.logical_op_id = logical_op_id
        self.op_id = None

    def __str__(self):
        op = f"{self.input_schema.class_name()} -> {self.op_name()} -> {self.output_schema.class_name()}\n"
        op += f"    ({', '.join(self.input_schema.field_names())[:30]}) "
        op += f"-> ({', '.join(self.output_schema.field_names())[:30]})\n"
        if getattr(self, "model", None):
            op += f"    Model: {self.model}\n"
        return op

    def get_copy_kwargs(self):
        """Return kwargs to assist sub-classes w/copy() calls."""
        return {
            "output_schema": self.output_schema,
            "input_schema": self.input_schema,
            "logical_op_id": self.logical_op_id,
            "max_workers": self.max_workers,
            "target_cache_id": self.target_cache_id,
            "verbose": self.verbose,
        }

    def op_name(self) -> str:
        """Name of the physical operator."""
        return str(self.__class__.__name__)

    def get_op_params(self):
        """
        You should implement get_op_params with op-specific parameters.
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

    def __eq__(self, other: PhysicalOperator) -> bool:
        raise NotImplementedError("Calling __eq__ on abstract method")

    def __hash__(self):
        return int(self.op_id, 16)

    def copy(self):
        copy_kwargs = self.get_copy_kwargs()
        return self.__class__(**copy_kwargs)

    def __call__(self, candidate: DataRecord) -> list[DataRecordsWithStats]:
        raise NotImplementedError("Calling __call__ from abstract method")

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - quality

        The function takes an argument which contains the OperatorCostEstimates
        of the physical operator whose output is the input to this operator.

        For the implemented operator. These will be used by the CostModel
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("CostEstimates from abstract method")
