from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class GenerationStats:
    """
    Dataclass for storing statistics about the execution of an operator on a single record.
    """

    model_name: str | None = None

    # The raw answer as output from the generator (a list of strings, possibly of len 1)
    # raw_answers: Optional[List[str]] = field(default_factory=list)

    # the total number of input tokens processed by this operator; None if this operation did not use an LLM
    # typed as a float because GenerationStats may be amortized (i.e. divided) across a number of output records
    total_input_tokens: float = 0.0

    # the total number of output tokens processed by this operator; None if this operation did not use an LLM
    # typed as a float because GenerationStats may be amortized (i.e. divided) across a number of output records
    total_output_tokens: float = 0.0

    # the total cost of processing the input tokens; None if this operation did not use an LLM
    total_input_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_output_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    cost_per_record: float = 0.0

    # (if applicable) the time (in seconds) spent executing a call to an LLM
    llm_call_duration_secs: float = 0.0

    # (if applicable) the time (in seconds) spent executing a call to a function
    fn_call_duration_secs: float = 0.0

    def __iadd__(self, other: GenerationStats) -> GenerationStats:
        # self.raw_answers.extend(other.raw_answers)
        for dataclass_field in [
            "total_input_tokens",
            "total_output_tokens",
            "total_input_cost",
            "total_output_cost",
            "cost_per_record",
            "llm_call_duration_secs",
            "fn_call_duration_secs",
        ]:
            setattr(self, dataclass_field, getattr(self, dataclass_field) + getattr(other, dataclass_field))
        return self

    def __add__(self, other: GenerationStats) -> GenerationStats:
        dct = {
            field: getattr(self, field) + getattr(other, field)
            for field in [
                "total_input_tokens",
                "total_output_tokens",
                "total_input_cost",
                "total_output_cost",
                "llm_call_duration_secs",
                "fn_call_duration_secs",
                "cost_per_record",
            ]
        }
        # dct['raw_answers'] = self.raw_answers + other.raw_answers
        dct["model_name"] = self.model_name
        return GenerationStats(**dct)

    # Do the same as iadd and add but with division operator
    def __itruediv__(self, quotient: float) -> GenerationStats:
        if quotient == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        if isinstance(quotient, int):
            quotient = float(quotient)
        for dataclass_field in [
            "total_input_tokens",
            "total_output_tokens",
            "total_input_cost",
            "total_output_cost",
            "cost_per_record",
            "llm_call_duration_secs",
            "fn_call_duration_secs",
        ]:
            setattr(self, dataclass_field, getattr(self, dataclass_field) / quotient)
        return self

    def __truediv__(self, quotient: float) -> GenerationStats:
        if quotient == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        if isinstance(quotient, int):
            quotient = float(quotient)
        dct = {
            field: getattr(self, field) / quotient
            for field in [
                "total_input_tokens",
                "total_output_tokens",
                "total_input_cost",
                "total_output_cost",
                "llm_call_duration_secs",
                "fn_call_duration_secs",
                "cost_per_record",
            ]
        }
        dct["model_name"] = self.model_name
        return GenerationStats(**dct)

    def __radd__(self, other: int) -> GenerationStats:
        return self


@dataclass
class RecordOpStats:
    """
    Dataclass for storing statistics about the execution of an operator on a single record.
    """

    ##### REQUIRED FIELDS #####
    # record id; an identifier for this record
    record_id: str

    # identifier for the parent of this record
    record_parent_id: str

    # idenifier for the source of this record
    record_source_id: str

    # a dictionary with the record state after being processed by the operator
    record_state: dict[str, Any]

    # operation id; an identifier for this operation's physical op id
    op_id: str

    # logical operation id; the logical op id for this physical op
    logical_op_id: str

    # operation name
    op_name: str

    # the time spent by the data record just in this operation
    time_per_record: float

    # the cost (in dollars) to generate this record at this operation
    cost_per_record: float

    ##### NOT-OPTIONAL, BUT FILLED BY EXECUTION CLASS AFTER CONSTRUCTOR CALL #####
    # the ID of the physical operation which produced the input record for this record at this operation
    source_op_id: str | None = None

    # the ID of the physical plan which produced this record at this operation
    plan_id: str = ""

    ##### OPTIONAL, BUT FILLED BY COST MODEL AFTER SAMPLE DATA EXECUTION #####
    quality: float | None = None

    ##### OPTIONAL FIELDS (I.E. ONLY MANDATORY FOR CERTAIN OPERATORS) #####
    # (if applicable) the name of the model used to generate the output for this record
    model_name: str | None = None

    # (if applicable) the mapping from field-name to generated output for this record
    answer: dict[str, Any] | None = None

    # (if applicable) the mapping from field-name to generated output for this record
    # raw_answers: Optional[List[str, Any]] = field(default_factory=list)

    # (if applicable) the list of input fields for the generation for this record
    input_fields: list[str] | None = None

    # (if applicable) the list of generated fields for this record
    generated_fields: list[str] | None = None

    # the total number of input tokens processed by this operator; None if this operation did not use an LLM
    # typed as a float because GenerationStats may be amortized (i.e. divided) across a number of output records
    total_input_tokens: float = 0.0

    # the total number of output tokens processed by this operator; None if this operation did not use an LLM
    # typed as a float because GenerationStats may be amortized (i.e. divided) across a number of output records
    total_output_tokens: float = 0.0

    # the total cost of processing the input tokens; None if this operation did not use an LLM
    total_input_cost: float = 0.0

    # the total cost of processing the output tokens; None if this operation did not use an LLM
    total_output_cost: float = 0.0

    # (if applicable) the filter text (or a string representation of the filter function) applied to this record
    filter_str: str | None = None

    # the True/False result of whether this record was output by the operator or not
    # (can only be False if the operator is as Filter)
    passed_operator: bool = True

    # (if applicable) the time (in seconds) spent executing a call to an LLM
    llm_call_duration_secs: float = 0.0

    # (if applicable) the time (in seconds) spent executing a UDF or calling an external api
    fn_call_duration_secs: float = 0.0

    # (if applicable) a boolean indicating whether this is the statistics captured from a failed convert operation
    failed_convert: bool | None = None

    # (if applicable) a boolean indicating whether this is an image convert/filter operation
    image_operation: bool | None = None

    # an OPTIONAL dictionary with more detailed information about this operation;
    op_details: dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class OperatorStats:
    """
    Dataclass for storing statistics captured within a given operator.
    """

    # the ID of the physical operation in which these stats were collected
    op_id: str

    # the name of the physical operation in which these stats were collected
    op_name: str

    # the total time spent in this operation
    total_op_time: float = 0.0

    # the total cost of this operation
    total_op_cost: float = 0.0

    # a list of RecordOpStats processed by the operation
    record_op_stats_lst: list[RecordOpStats] = field(default_factory=list)

    # an OPTIONAL dictionary with more detailed information about this operation;
    op_details: dict[str, Any] = field(default_factory=dict)

    def add_record_op_stats(
        self,
        record_op_stats_lst: RecordOpStats | list[RecordOpStats],
        source_op_id: str | None,
        plan_id: str,
    ):
        # convert individual record into list
        if not isinstance(record_op_stats_lst, list):
            record_op_stats_lst = [record_op_stats_lst]

        # update op stats
        for record_op_stats in record_op_stats_lst:
            record_op_stats.source_op_id = source_op_id
            record_op_stats.plan_id = plan_id
            self.record_op_stats_lst.append(record_op_stats)
            self.total_op_time += record_op_stats.time_per_record
            self.total_op_cost += record_op_stats.cost_per_record

    def __iadd__(self, op_stats: OperatorStats):
        """NOTE: we assume the execution layer guarantees these op_stats belong to the same operator."""
        self.total_op_time += op_stats.total_op_time
        self.total_op_cost += op_stats.total_op_cost
        self.record_op_stats_lst.extend(op_stats.record_op_stats_lst)
        return self

    def to_json(self):
        return {
            "op_id": self.op_id,
            "op_name": self.op_name,
            "total_op_time": self.total_op_time,
            "total_op_cost": self.total_op_cost,
            "record_op_stats_lst": [record_op_stats.to_json() for record_op_stats in self.record_op_stats_lst],
            "op_details": self.op_details,
        }


@dataclass
class PlanStats:
    """
    Dataclass for storing statistics captured for an entire plan.
    """

    # id for identifying the physical plan
    plan_id: str

    # string representation of the physical plan
    plan_str: str | None = None

    # dictionary of OperatorStats objects (one for each operator)
    operator_stats: dict[str, OperatorStats] = field(default_factory=dict)

    # total runtime for the plan measured from the start to the end of PhysicalPlan.execute()
    total_plan_time: float = 0.0

    # total cost for plan
    total_plan_cost: float = 0.0

    def __iadd__(self, plan_stats: PlanStats):
        """
        NOTE: we assume the execution layer guarantees:
        1. these plan_stats belong to the same plan
        2. these plan_stats come from sequential (non-overlapping) executions of the same plan

        The latter criteria implies it is okay for this method to sum the plan (and operator) runtimes.
        """
        self.total_plan_time += plan_stats.total_plan_time
        self.total_plan_cost += plan_stats.total_plan_cost
        for op, op_stats in plan_stats.operator_stats.items():
            if op in self.operator_stats:
                self.operator_stats[op] += op_stats
            else:
                self.operator_stats[op] = op_stats

    def finalize(self, total_plan_time: float):
        self.total_plan_time = total_plan_time
        self.total_plan_cost = sum([op_stats.total_op_cost for _, op_stats in self.operator_stats.items()])

    def __str__(self):
        stats = f"Total_plan_time={self.total_plan_time} \n"
        stats += f"Total_plan_cost={self.total_plan_cost} \n"
        for idx, op_stats in enumerate(self.operator_stats.values()):
            stats += f"{idx}. {op_stats.op_name} time={op_stats.total_op_time} cost={op_stats.total_op_cost} \n"
        return stats

    def to_json(self):
        return {
            "plan_id": self.plan_id,
            "plan_str": self.plan_str,
            "operator_stats": {op_id: op_stats.to_json() for op_id, op_stats in self.operator_stats.items()},
            "total_plan_time": self.total_plan_time,
            "total_plan_cost": self.total_plan_cost,
        }


@dataclass
class ExecutionStats:
    """
    Dataclass for storing statistics captured for the entire execution of a workload.
    """

    # string for identifying this workload execution
    execution_id: str | None = None

    # dictionary of PlanStats objects (one for each plan run during execution)
    plan_stats: dict[str, PlanStats] = field(default_factory=dict)

    # total time spent optimizing
    total_optimization_time: float = 0.0

    # total runtime for a plan's execution
    total_execution_time: float = 0.0

    # total cost for a plan's execution
    total_execution_cost: float = 0.0

    # dictionary of plan strings; useful for printing executed plans in demos
    plan_strs: dict[str, str] = field(default_factory=dict)

    def to_json(self):
        return {
            "execution_id": self.execution_id,
            "plan_stats": {plan_id: plan_stats.to_json() for plan_id, plan_stats in self.plan_stats.items()},
            "total_optimization_time": self.total_optimization_time,
            "total_execution_time": self.total_execution_time,
            "total_execution_cost": self.total_execution_cost,
            "plan_strs": self.plan_strs,
        }


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

    # lower bound on cardinality
    cardinality_lower_bound: float | None = None

    # upper bound on cardinality
    cardinality_upper_bound: float | None = None

    # lower bound on time_per_record
    time_per_record_lower_bound: float | None = None

    # upper bound on time_per_record
    time_per_record_upper_bound: float | None = None

    # lower bound on cost_per_record
    cost_per_record_lower_bound: float | None = None

    # upper bound on cost_per_record
    cost_per_record_upper_bound: float | None = None

    # lower bound on quality
    quality_lower_bound: float | None = None

    # upper bound on quality
    quality_upper_bound: float | None = None

    def __post_init__(self):
        if self.cardinality_lower_bound is None and self.cardinality_upper_bound is None:
            self.cardinality_lower_bound = self.cardinality
            self.cardinality_upper_bound = self.cardinality

        if self.time_per_record_lower_bound is None and self.time_per_record_upper_bound is None:
            self.time_per_record_lower_bound = self.time_per_record
            self.time_per_record_upper_bound = self.time_per_record

        if self.cost_per_record_lower_bound is None and self.cost_per_record_upper_bound is None:
            self.cost_per_record_lower_bound = self.cost_per_record
            self.cost_per_record_upper_bound = self.cost_per_record

        if self.quality_lower_bound is None and self.quality_upper_bound is None:
            self.quality_lower_bound = self.quality
            self.quality_upper_bound = self.quality


@dataclass
class PlanCost:
    """
    Dataclass for storing the (cost, time, quality) estimates of (sub)-plans and their upper and lower bounds.
    """

    # the expression cost
    cost: float

    # the expression runtime
    time: float

    # the expression quality
    quality: float

    # operator-specific cost estimates
    op_estimates: OperatorCostEstimates | None = None

    # lower bound on the expression cost
    cost_lower_bound: float | None = None

    # upper bound on the expression cost
    cost_upper_bound: float | None = None

    # lower bound on the expression time
    time_lower_bound: float | None = None

    # upper bound on the expression time
    time_upper_bound: float | None = None

    # lower bound on the expression quality
    quality_lower_bound: float | None = None

    # upper bound on the expression quality
    quality_upper_bound: float | None = None

    def __hash__(self):
        return hash(f"{self.cost}-{self.time}-{self.quality}")

    def __post_init__(self):
        if self.time_lower_bound is None and self.time_upper_bound is None:
            self.time_lower_bound = self.time
            self.time_upper_bound = self.time

        if self.cost_lower_bound is None and self.cost_upper_bound is None:
            self.cost_lower_bound = self.cost
            self.cost_upper_bound = self.cost

        if self.quality_lower_bound is None and self.quality_upper_bound is None:
            self.quality_lower_bound = self.quality
            self.quality_upper_bound = self.quality

    def __iadd__(self, other: PlanCost) -> PlanCost:
        """
        NOTE: we currently assume the updating of the op_estimates are handled by the caller
        as there is not a universally correct meaning of addition of op_estiamtes.
        """
        self.cost += other.cost
        self.time += other.time
        self.quality *= other.quality
        for dataclass_field in ["cost_lower_bound", "cost_upper_bound", "time_lower_bound", "time_upper_bound"]:
            if getattr(self, dataclass_field) is not None and getattr(other, dataclass_field) is not None:
                summation = getattr(self, dataclass_field) + getattr(other, dataclass_field)
                setattr(self, dataclass_field, summation)

        for dataclass_field in ["quality_lower_bound", "quality_upper_bound"]:
            if getattr(self, dataclass_field) is not None and getattr(other, dataclass_field) is not None:
                product = getattr(self, dataclass_field) * getattr(other, dataclass_field)
                setattr(self, dataclass_field, product)

        return self

    def __add__(self, other: PlanCost) -> PlanCost:
        """
        NOTE: we currently assume the updating of the op_estimates are handled by the caller
        as there is not a universally correct meaning of addition of op_estiamtes.
        """
        dct = {
            field: getattr(self, field) + getattr(other, field)
            for field in [
                "cost",
                "cost_lower_bound",
                "cost_upper_bound",
                "time",
                "time_lower_bound",
                "time_upper_bound",
            ]
        }
        for dataclass_field in ["quality", "quality_lower_bound", "quality_upper_bound"]:
            dct[dataclass_field] = getattr(self, dataclass_field) * getattr(other, dataclass_field)

        return PlanCost(**dct)
