from __future__ import annotations

import json

from palimpzest.core.data.dataclasses import OperatorCostEstimates
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Schema
from palimpzest.datamanager.datamanager import DataDirectory
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
        self.datadir = DataDirectory()

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
        NOTE: input_schema is not included in the id params because it depends on how the Optimizer orders operations.
        """
        return {"output_schema": self.output_schema}

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
        id_params = {k: str(v) for k, v in id_params.items()}

        # compute, set, and return the op_id
        hash_str = json.dumps({"op_name": op_name, **id_params}, sort_keys=True)
        self.op_id = hash_for_id(hash_str)

        return self.op_id

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

    def get_fields_to_generate(self, candidate: DataRecord, input_schema: Schema, output_schema: Schema) -> list[str]:
        """
        Creates the list of field names that an operation needs to generate. Right now this is only used
        by convert and retrieve operators.
        """
        # construct the list of fields in output_schema which will need to be generated;
        # specifically, this is the set of fields which are:
        # 1. not declared in the input schema, and
        # 2. not present in the candidate's attributes
        #    a. if the field is present, but its value is None --> we will try to generate it
        fields_to_generate = []
        for field_name in output_schema.field_names():
            if field_name not in input_schema.field_names() and getattr(candidate, field_name, None) is None:
                fields_to_generate.append(field_name)

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
        execution data alone -- thus DataSourcePhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("CostEstimates from abstract method")

    def __call__(self, candidate: DataRecord) -> DataRecordSet:
        raise NotImplementedError("Calling __call__ from abstract method")

    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: DataRecord | list[DataRecord]) -> tuple[DataRecordSet, PhysicalOperator]:
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        record_set = operator(op_input)

        return record_set, operator, op_input
    