from __future__ import annotations

from palimpzest.constants import MAX_OP_ID_CHARS
from palimpzest.corelib import Schema
from palimpzest.dataclasses import RecordOpStats, OperatorCostEstimates
from palimpzest.datamanager import DataDirectory
from palimpzest.elements import DataRecord

from typing import List, Tuple, Optional

import hashlib
import json

# TYPE DEFINITIONS
DataRecordsWithStats = Tuple[List[DataRecord], List[RecordOpStats]]


class PhysicalOperator():
    """
    All implemented physical operators should inherit from this class, and define in the implemented_op variable
    exactly which logical operator they implement. This is necessary for the planner to be able to determine
    which physical operators can be used to implement a given logical operator.
    """

    LOCAL_PLAN = "LOCAL"
    REMOTE_PLAN = "REMOTE"

    implemented_op = None
    inputSchema = None
    outputSchema = None
    final = False # This gets set to True if the operator actually is to be used

    @classmethod
    def implements(cls, logical_operator_class):
        return logical_operator_class == cls.implemented_op and cls.final

    def __init__(
        self,
        outputSchema: Schema,
        inputSchema: Optional[Schema] = None,
        shouldProfile: bool = False,
        max_workers: int = 1,
        *args, **kwargs
    ) -> None:
        self.outputSchema = outputSchema
        self.inputSchema = inputSchema
        self.datadir = DataDirectory()
        self.shouldProfile = shouldProfile
        self.max_workers = max_workers

    def __eq__(self, other: PhysicalOperator) -> bool:
        raise NotImplementedError("Calling __eq__ on abstract method")

    def op_name(self) -> str:
        """Name of the physical operator."""
        return self.__class__.__name__

    def get_op_dict(self):
        raise NotImplementedError("You should implement get_op_dict with op specific parameters")
    
    def get_op_id(self, plan_position: Optional[int] = None) -> str:
        op_dict = self.get_op_dict()
        if plan_position is not None:
            op_dict["plan_position"] = plan_position

        ordered = json.dumps(op_dict, sort_keys=True)
        hash = hashlib.sha256(ordered.encode()).hexdigest()[:MAX_OP_ID_CHARS]

        op_id = (
            f"{self.op_name()}_{hash}"
            if plan_position is None
            else f"{self.op_name()}_{plan_position}_{hash}"
        )
        return op_id

    def is_hardcoded(self) -> bool:
        """ By default, operators are not hardcoded.
        In those that implement HardcodedConvert or HardcodedFilter, this will return True."""
        return False
        

    def copy(self) -> PhysicalOperator:
        raise NotImplementedError("__copy___ on abstract class")

    def __call__(self, candidate: DataRecord) -> List[DataRecordsWithStats]:
        raise NotImplementedError("Using __call__ from abstract method")

    def naiveCostEstimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        This function returns a naive estimate of this operator's:
        - cardinality
        - time_per_record
        - cost_per_record
        - quality

        The function takes an argument which contains the OperatorCostEstimates
        of the physical operator whose output is the input to this operator.
    
        For the implemented operator. These will be used by the CostEstimator
        when PZ does not have sample execution data -- and it will be necessary
        in some cases even when sample execution data is present. (For example,
        the cardinality of each operator cannot be estimated based on sample
        execution data alone -- thus DataSourcePhysicalOps need to give
        at least ballpark correct estimates of this quantity).
        """
        raise NotImplementedError("CostEstimates from abstract method")
