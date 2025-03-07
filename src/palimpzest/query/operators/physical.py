from __future__ import annotations

import json

from palimpzest.core.data.dataclasses import OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Schema
from palimpzest.utils.hash_helpers import hash_for_id


class PhysicalOperator:
    """
    All implemented physical operators should inherit from this class.
    In order for the Optimizer to consider using a physical operator for a
    given logical operation, the user must also write an ImplementationRule.
    """

    def __init__(
        self,
        output_schema: Schema,
        input_schema: Schema | None = None,
        depends_on: list[str] | None = None,
        logical_op_id: str | None = None,
        logical_op_name: str | None = None,
        target_cache_id: str | None = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.depends_on = depends_on if depends_on is None else sorted(depends_on)
        self.logical_op_id = logical_op_id
        self.logical_op_name = logical_op_name
        self.target_cache_id = target_cache_id
        self.verbose = verbose
        self.op_id = None

        # compute the fields generated by this physical operator
        input_field_names = self.input_schema.field_names() if self.input_schema is not None else []
        self.generated_fields = sorted([
            field_name
            for field_name in self.output_schema.field_names()
            if field_name not in input_field_names
        ])

        # sets __hash__() for each child Operator to be the base class' __hash__() method;
        # by default, if a subclass defines __eq__() but not __hash__() Python will set that
        # class' __hash__ to None
        self.__class__.__hash__ = PhysicalOperator.__hash__

    def __str__(self):
        op = f"{self.input_schema.class_name()} -> {self.op_name()} -> {self.output_schema.class_name()}\n"
        op += f"    ({', '.join(self.input_schema.field_names())[:30]}) "
        op += f"-> ({', '.join(self.output_schema.field_names())[:30]})\n"
        if getattr(self, "model", None):
            op += f"    Model: {self.model}\n"
        return op

    def __eq__(self, other) -> bool:
        all_id_params_match = all(value == getattr(other, key) for key, value in self.get_id_params().items())
        return isinstance(other, self.__class__) and all_id_params_match

    def copy(self) -> PhysicalOperator:
        return self.__class__(**self.get_op_params())

    def op_name(self) -> str:
        """Name of the physical operator."""
        return str(self.__class__.__name__)

    def get_id_params(self) -> dict:
        """
        Returns a dictionary mapping of physical operator parameters which are relevant
        for computing the physical operator id.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        NOTE: input_schema and output_schema are not included in the id params by default,
              because they may depend on the order of operations chosen by the Optimizer.
              This is particularly true for convert operations, where the output schema
              is now the union of the input and output schemas of the logical operator.
        """
        return {"generated_fields": self.generated_fields}

    def get_op_params(self) -> dict:
        """
        Returns a dictionary mapping of physical operator parameters which may be used to
        create a copy of this physical operation.

        NOTE: Should be overriden by subclasses to include class-specific parameters.
        """
        return {
            "output_schema": self.output_schema,
            "input_schema": self.input_schema,
            "depends_on": self.depends_on,
            "logical_op_id": self.logical_op_id,
            "logical_op_name": self.logical_op_name,
            "target_cache_id": self.target_cache_id,
            "verbose": self.verbose,
        }

    def get_op_id(self):
        """
        NOTE: We do not call this in the __init__() method as subclasses may set parameters
              returned by self.get_id_params() after they call to super().__init__().

        NOTE: This is NOT a universal ID.

        Two different PhysicalOperator instances with the identical returned values
        from the call to self.get_id_params() will have equivalent op_ids.
        """
        # return self.op_id if we've computed it before
        if self.op_id is not None:
            return self.op_id

        # get op name and op parameters which are relevant for computing the id
        op_name = self.op_name()
        id_params = self.get_id_params()
        id_params = {
            k: str(v) if k != "output_schema" else sorted(v.field_names())
            for k, v in id_params.items()
        }

        # compute, set, and return the op_id
        hash_str = json.dumps({"op_name": op_name, **id_params}, sort_keys=True)
        self.op_id = hash_for_id(hash_str)

        return self.op_id
    
    def get_logical_op_id(self) -> str | None:
        return self.logical_op_id

    def __hash__(self):
        return int(self.op_id, 16)

    def get_model_name(self) -> str | None:
        """Returns the name of the model used by the physical operator (if it sets self.model). Otherwise, it returns None."""
        return None

    def get_input_fields(self):
        """Returns the set of input fields needed to execute a physical operator."""
        depends_on_fields = (
            [field.split(".")[-1] for field in self.depends_on]
            if self.depends_on is not None and len(self.depends_on) > 0
            else None
        )
        input_fields = (
            self.input_schema.field_names()
            if depends_on_fields is None
            else [field for field in self.input_schema.field_names() if field in depends_on_fields]
        )

        return input_fields

    def get_fields_to_generate(self, candidate: DataRecord) -> list[str]:
        """
        Returns the list of field names that this operator needs to generate for the given candidate.
        This function returns only the fields in self.generated_fields which are not already present
        in the candidate. This is important for operators with retry logic, where we may only need to
        recompute a subset of self.generated_fields.

        Right now this is only used by convert and retrieve operators.
        """
        fields_to_generate = [
            field_name
            for field_name in self.generated_fields
            if getattr(candidate, field_name, None) is None
        ]

        return fields_to_generate

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
        execution data alone -- thus ScanPhysicalOps need to give at least ballpark
        correct estimates of this quantity).
        """
        raise NotImplementedError("CostEstimates from abstract method")

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        raise NotImplementedError("Calling __call__ from abstract method")
