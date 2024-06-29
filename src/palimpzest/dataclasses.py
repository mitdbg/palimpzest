from __future__ import annotations

from palimpzest.elements import DataRecord
from dataclasses import dataclass, asdict, field

from typing import Any, Dict, List, Optional, Union

@dataclass
class GenerationStats:
    """
    Dataclass for storing statistics about the execution of an operator on a single record.
    """
    model_name: Optional[str] = None

    # The raw answer as output from the generator (a list of strings, possibly of len 1)
    # raw_answers: Optional[List[str]] = field(default_factory=list)
    
    # the total number of input tokens processed by this operator; None if this operation did not use an LLM
    total_input_tokens: int = 0.0

    # the total number of output tokens processed by this operator; None if this operation did not use an LLM
    total_output_tokens: int = 0.0

    # the total cost of processing the input tokens; None if this operation did not use an LLM
    total_input_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_output_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_cost: float = 0.0

    # (if applicable) the time (in seconds) spent executing a call to an LLM
    llm_call_duration_secs: float = 0.0

    # (if applicable) the time (in seconds) spent executing a call to a function
    fn_call_duration_secs: float = 0.0

    def __iadd__(self, other: GenerationStats) -> GenerationStats:
#        self.raw_answers.extend(other.raw_answers)
        for field in ['total_input_tokens', 'total_output_tokens', 'total_input_cost', 'total_output_cost','total_cost','llm_call_duration_secs', 'fn_call_duration_secs']:
            setattr(self, field, getattr(self, field) + getattr(other, field))
        return self

    def __add__(self, other: GenerationStats) -> GenerationStats:
        dct = {field: getattr(self, field) + getattr(other, field) for field in ['total_input_tokens', 'total_output_tokens', 'total_input_cost', 'total_output_cost', 'llm_call_duration_secs', 'fn_call_duration_secs', 'total_cost']}
        # dct['raw_answers'] = self.raw_answers + other.raw_answers
        dct['model_name'] = self.model_name      
        return GenerationStats(**dct)
    
    # Do the same as iadd and add but with division operator
    def __itruediv__(self, quotient: float) -> GenerationStats:
        if quotient == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        if isinstance(quotient, int):
            quotient = float(quotient)
        for field in ['total_input_tokens', 'total_output_tokens', 'total_input_cost', 'total_output_cost','total_cost','llm_call_duration_secs', 'fn_call_duration_secs']:
            setattr(self, field, getattr(self, field) / quotient)
        return self
    
    def __truediv__(self, quotient: float) -> GenerationStats:
        if quotient == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        if isinstance(quotient, int):
            quotient = float(quotient)
        dct = {field: getattr(self, field) / quotient for field in ['total_input_tokens', 'total_output_tokens', 'total_input_cost', 'total_output_cost', 'llm_call_duration_secs', 'fn_call_duration_secs', 'total_cost']}
        dct['model_name'] = self.model_name      
        return GenerationStats(**dct)
    
@dataclass
class RecordOpStats:
    """
    Dataclass for storing statistics about the execution of an operator on a single record.
    """
    ##### REQUIRED FIELDS #####
    # record id; a unique identifier for this record
    record_uuid: str

    # unique identifier for the parent of this record
    record_parent_uuid: str

    # a dictionary with the record state after being processed by the operator
    record_state: Dict[str, Any]

    # operation id; a unique identifier for this operation
    op_id: str

    # operation name
    op_name: str

    # the time spent by the data record just in this operation
    time_per_record: float

    # the cost (in dollars) to generate this record at this operation
    cost_per_record: float

    ##### NOT-OPTIONAL, BUT FILLED BY EXECUTION CLASS AFTER CONSTRUCTOR CALL #####
    # the ID of the physical operation which produced the input record for this record at this operation
    source_op_id: str = ""

    ##### OPTIONAL FIELDS (I.E. ONLY MANDATORY FOR CERTAIN OPERATORS) #####
    # (if applicable) the name of the model used to generate the output for this record
    model_name: Optional[str] = None

    # (if applicable) the mapping from field-name to generated output for this record
    answer: Optional[Dict[str, Any]] = None

    # (if applicable) the mapping from field-name to generated output for this record
    # raw_answers: Optional[List[str, Any]] = field(default_factory=list)

    # (if applicable) the list of input fields for the generation for this record
    input_fields: Optional[List[str]] = None

    # (if applicable) the list of generated fields for this record
    generated_fields: Optional[List[str]] = None

    # the total number of input tokens processed by this operator; None if this operation did not use an LLM
    total_input_tokens: int = 0.0

    # the total number of output tokens processed by this operator; None if this operation did not use an LLM
    total_output_tokens: int = 0.0

    # the total cost of processing the input tokens; None if this operation did not use an LLM
    total_input_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_output_cost: float = 0.0

    # (if applicable) the filter text (or a string representation of the filter function) applied to this record
    filter_str: Optional[str] = None

    # (if applicable) the True/False result of whether this record passed the filter or not
    passed_filter: Optional[bool] = None

    # (if applicable) the time (in seconds) spent executing a call to an LLM
    llm_call_duration_secs: float = 0.0

    # (if applicable) the time (in seconds) spent executing a UDF or calling an external api
    fn_call_duration_secs: float = 0.0

    # @staticmethod
    # def from_record_and_kwargs(record: DataRecord, **kwargs: Dict[str, Any]) -> RecordOpStats:
    #     return RecordOpStats(
    #         record_uuid=record._uuid,
    #         record_parent_uuid=record._parent_uuid,
    #         op_id=kwargs['op_id'],
    #         op_name=kwargs['op_name'],
    #         time_per_record=kwargs['time_per_record'],
    #         cost_per_record=kwargs['cost_per_record'],
    #         record_state=record._asDict(include_bytes=False),
    #         record_details=kwargs.get('record_details', None),
    #     )

    # def to_dict(self):
    #     return asdict(self)

@dataclass
class OperatorStats:
    """
    Dataclass for storing statistics captured within a given operator.
    """
    # the index of the operator in the plan
    op_idx: int

    # the ID of the physical operation in which these stats were collected
    op_id: str

    # the name of the physical operation in which these stats were collected
    op_name: str

    # the total time spent in this operation
    total_op_time: float = 0.0

    # the total cost of this operation
    total_op_cost: float = 0.0

    # a list of RecordOpStats processed by the operation
    record_op_stats_lst: List[RecordOpStats] = field(default_factory=list)

    # an OPTIONAL dictionary with more detailed information about this operation;
    op_details: Dict[str, Any] = field(default_factory=dict)

    def __iadd__(self, record_op_stats: RecordOpStats):
        self.total_op_time += record_op_stats.time_per_record
        self.total_op_cost += record_op_stats.cost_per_record
        self.record_op_stats_lst.append(record_op_stats)

    # def to_dict(self):
    #     return asdict(self)


@dataclass
class PlanStats:
    """
    Dataclass for storing statistics captured for an entire plan.
    """
    # string for identifying the physical plan
    plan_id: str

    # dictionary of OperatorStats objects (one for each operator)
    operator_stats: Dict[str, OperatorStats] = field(default_factory=dict)

    # total runtime for the plan measured from the start to the end of PhysicalPlan.execute()
    total_plan_time: float = 0.0

    # total cost for plan
    total_plan_cost: float = 0.0

    def finalize(self, total_plan_time: float):
        self.total_plan_time = total_plan_time
        self.total_plan_cost = sum([op_stats.total_op_cost for _, op_stats in self.operator_stats.items()])


@dataclass
class ExecutionStats:
    """
    Dataclass for storing statistics captured for the entire execution of a workload.
    """
    # string for identifying this workload execution
    execution_id: str = None

    # dictionary of PlanStats objects (one for each plan run during execution)
    plan_stats: Dict[str, PlanStats] = field(default_factory=dict)

    # total runtime for a call to pz.Execute
    total_execution_time: float = 0.0

    # total cost for a call to pz.Execute
    total_execution_cost: float = 0.0


@dataclass
class OperatorCostEstimates:
    """
    Dataclass for storing estimates of key metrics of interest for each operator.
    """
    # (estimated) number of records output by this operator
    cardinality: float

    # (estimated) avg. time spent in this operator per-record
    time_per_record: float

    # (estimated) dollars spent per-record by this operator
    cost_per_record: float

    # (estimated) quality of the output from this operator
    quality: float
