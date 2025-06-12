from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np


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

    # (if applicable) the total number of LLM calls made by this operator
    total_llm_calls: int = 0

    # (if applicable) the total number of embedding LLM calls made by this operator
    total_embedding_llm_calls: int = 0

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
            "total_llm_calls",
            "total_embedding_llm_calls",
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
                "total_llm_calls",
                "total_embedding_llm_calls",
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
            "total_llm_calls",
            "total_embedding_llm_calls",
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
                "total_llm_calls",
                "total_embedding_llm_calls",
                "cost_per_record",
            ]
        }
        dct["model_name"] = self.model_name
        return GenerationStats(**dct)

    def __radd__(self, other: int) -> GenerationStats:
        assert not isinstance(other, GenerationStats), "This should not be called with a GenerationStats object"
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

    # idenifier for the source idx of this record
    record_source_idx: str

    # a dictionary with the record state after being processed by the operator
    record_state: dict[str, Any]

    # operation id; an identifier for this operation's physical op id
    full_op_id: str

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
    source_full_op_id: str | None = None

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

    # (if applicable) the total number of LLM calls made by this operator
    total_llm_calls: int = 0

    # (if applicable) the total number of embedding LLM calls made by this operator
    total_embedding_llm_calls: int = 0

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

    # the full ID of the physical operation in which these stats were collected
    full_op_id: str

    # the name of the physical operation in which these stats were collected
    op_name: str

    # the total time spent in this operation
    total_op_time: float = 0.0

    # the total cost of this operation
    total_op_cost: float = 0.0

    # a list of RecordOpStats processed by the operation
    record_op_stats_lst: list[RecordOpStats] = field(default_factory=list)

    # the full ID of the physical operator which precedes this one
    source_full_op_id: str | None = None

    # the ID of the physical plan which this operator is part of
    plan_id: str = ""

    # an OPTIONAL dictionary with more detailed information about this operation;
    op_details: dict[str, Any] = field(default_factory=dict)

    def __iadd__(self, stats: OperatorStats | RecordOpStats) -> OperatorStats:
        """
        Sum the given stats to this operator's stats. The given stats can be either:

        1. an OperatorStats object
        2. a RecordOpStats object

        NOTE: in case (1.) we assume the execution layer guarantees that `stats` is
              generated by the same operator in the same plan. Thus, we assume the
              full_op_ids, op_name, source_op_id, etc. do not need to be updated.
        """
        if isinstance(stats, OperatorStats):
            self.total_op_time += stats.total_op_time
            self.total_op_cost += stats.total_op_cost
            self.record_op_stats_lst.extend(stats.record_op_stats_lst)

        elif isinstance(stats, RecordOpStats):
            stats.source_full_op_id = self.source_full_op_id
            stats.plan_id = self.plan_id
            self.record_op_stats_lst.append(stats)
            self.total_op_time += stats.time_per_record
            self.total_op_cost += stats.cost_per_record

        else:
            raise TypeError(f"Cannot add {type(stats)} to OperatorStats")

        return self

    def to_json(self):
        return {
            "full_op_id": self.full_op_id,
            "op_name": self.op_name,
            "total_op_time": self.total_op_time,
            "total_op_cost": self.total_op_cost,
            "record_op_stats_lst": [record_op_stats.to_json() for record_op_stats in self.record_op_stats_lst],
            "op_details": self.op_details,
        }

@dataclass
class BasePlanStats:
    """
    Dataclass for storing statistics captured for an entire plan.

    This class is subclassed for tracking:
    - PlanStats: the statistics for execution of a PhysicalPlan
    - SentinelPlanStats: the statistics for execution of a SentinelPlan

    The key difference between the two subclasses is that the `operator_stats`
    field in the PlanStats maps from the physical operator ids to their corresponding
    OperatorStats objects.

    The `operator_stats` field in the SentinelPlanStats maps from a logical operator id
    to another dictionary which maps from the physical operator ids to their corresponding
    OperatorStats objects.
    """

    # id for identifying the physical plan
    plan_id: str

    # string representation of the physical plan
    plan_str: str | None = None

    # dictionary whose values are OperatorStats objects;
    # PlanStats maps {full_op_id -> OperatorStats}
    # SentinelPlanStats maps {logical_op_id -> {full_op_id -> OperatorStats}}
    operator_stats: dict = field(default_factory=dict)

    # total runtime for the plan measured from the start to the end of PhysicalPlan.execute()
    total_plan_time: float = 0.0

    # total cost for plan
    total_plan_cost: float = 0.0

    # start time for the plan execution; should be set by calling PlanStats.start()
    start_time: float | None = None

    def start(self) -> None:
        """Start the timer for this plan execution."""
        self.start_time = time.time()

    def finish(self) -> None:
        """Finish the timer for this plan execution."""
        if self.start_time is None:
            raise RuntimeError("PlanStats.start() must be called before PlanStats.finish()")
        self.total_plan_time = time.time() - self.start_time
        self.total_plan_cost = self.sum_op_costs()

    @staticmethod
    @abstractmethod
    def from_plan(plan) -> BasePlanStats:
        """
        Initialize this PlanStats object from a PhysicalPlan or SentinelPlan object.
        """
        pass

    @abstractmethod
    def sum_op_costs(self) -> float:
        """
        Sum the costs of all operators in this plan.
        """
        pass

    @abstractmethod
    def add_record_op_stats(self, record_op_stats: RecordOpStats | list[RecordOpStats]) -> None:
        """
        Add the given RecordOpStats to this plan's operator stats.
        """
        pass

    @abstractmethod
    def __iadd__(self, plan_stats: BasePlanStats) -> None:
        """
        Add the given PlanStats to this plan's operator stats.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of this plan's statistics.
        """
        pass

    @abstractmethod
    def to_json(self) -> dict:
        """
        Return a JSON representation of this plan's statistics.
        """
        pass

@dataclass
class PlanStats(BasePlanStats):
    """
    Subclass of BasePlanStats which captures statistics from the execution of a single PhysicalPlan.
    """
    @staticmethod
    def from_plan(plan) -> PlanStats:
        """
        Initialize this PlanStats object from a PhysicalPlan object.
        """
        operator_stats = {}
        for op_idx, op in enumerate(plan.operators):
            full_op_id = op.get_full_op_id()
            operator_stats[full_op_id] = OperatorStats(
                full_op_id=full_op_id,
                op_name=op.op_name(),
                source_full_op_id=None if op_idx == 0 else plan.operators[op_idx - 1].get_full_op_id(),
                plan_id=plan.plan_id,
                op_details={k: str(v) for k, v in op.get_id_params().items()},
            )
    
        return PlanStats(plan_id=plan.plan_id, plan_str=str(plan), operator_stats=operator_stats)
 
    def sum_op_costs(self) -> float:
        """
        Sum the costs of all operators in this plan.
        """
        return sum([op_stats.total_op_cost for _, op_stats in self.operator_stats.items()])

    def add_record_op_stats(self, record_op_stats: RecordOpStats | list[RecordOpStats]) -> None:
        """
        Add the given RecordOpStats to this plan's operator stats.
        """
        # normalize input type to be list[RecordOpStats]
        record_op_stats_lst = record_op_stats if isinstance(record_op_stats, list) else [record_op_stats]

        # update operator stats
        for record_op_stats in record_op_stats_lst:
            full_op_id = record_op_stats.full_op_id
            if full_op_id in self.operator_stats:
                self.operator_stats[full_op_id] += record_op_stats
            else:
                raise ValueError(f"RecordOpStats with full_op_id {full_op_id} not found in PlanStats")

    def __iadd__(self, plan_stats: PlanStats) -> None:
        """
        NOTE: we assume the execution layer guarantees:
        1. these plan_stats belong to the same plan
        2. these plan_stats come from sequential (non-overlapping) executions of the same plan

        The latter criteria implies it is okay for this method to sum the plan (and operator) runtimes.
        """
        self.total_plan_time += plan_stats.total_plan_time
        self.total_plan_cost += plan_stats.total_plan_cost
        for full_op_id, op_stats in plan_stats.operator_stats.items():
            if full_op_id in self.operator_stats:
                self.operator_stats[full_op_id] += op_stats
            else:
                self.operator_stats[full_op_id] = op_stats

    def __str__(self) -> str:
        stats = f"total_plan_time={self.total_plan_time} \n"
        stats += f"total_plan_cost={self.total_plan_cost} \n"
        for idx, op_stats in enumerate(self.operator_stats.values()):
            stats += f"{idx}. {op_stats.op_name} time={op_stats.total_op_time} cost={op_stats.total_op_cost} \n"
        return stats

    def to_json(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "plan_str": self.plan_str,
            "operator_stats": {full_op_id: op_stats.to_json() for full_op_id, op_stats in self.operator_stats.items()},
            "total_plan_time": self.total_plan_time,
            "total_plan_cost": self.total_plan_cost,
        }


@dataclass
class SentinelPlanStats(BasePlanStats):
    """
    Subclass of BasePlanStats which captures statistics from the execution of a single SentinelPlan.
    """
    @staticmethod
    def from_plan(plan) -> SentinelPlanStats:
        """
        Initialize this PlanStats object from a Sentinel object.
        """
        operator_stats = {}
        for op_set_idx, (logical_op_id, op_set) in enumerate(plan):
            operator_stats[logical_op_id] = {}
            for physical_op in op_set:
                full_op_id = physical_op.get_full_op_id()
                operator_stats[logical_op_id][full_op_id] = OperatorStats(
                    full_op_id=full_op_id,
                    op_name=physical_op.op_name(),
                    source_full_op_id=None if op_set_idx == 0 else plan.logical_op_ids[op_set_idx - 1],  # NOTE: this may be a reason to keep `source_op_id` instead of `source_full_op_id`
                    plan_id=plan.plan_id,
                    op_details={k: str(v) for k, v in physical_op.get_id_params().items()},
                )
    
        return SentinelPlanStats(plan_id=plan.plan_id, plan_str=str(plan), operator_stats=operator_stats)

    def sum_op_costs(self) -> float:
        """
        Sum the costs of all operators in this plan.
        """
        return sum(sum([op_stats.total_op_cost for _, op_stats in phys_op_stats.items()]) for _, phys_op_stats in self.operator_stats.items())

    def add_record_op_stats(self, record_op_stats: RecordOpStats | list[RecordOpStats]) -> None:
        """
        Add the given RecordOpStats to this plan's operator stats.
        """
        # normalize input type to be list[RecordOpStats]
        record_op_stats_lst = record_op_stats if isinstance(record_op_stats, list) else [record_op_stats]

        # update operator stats
        for record_op_stats in record_op_stats_lst:
            logical_op_id = record_op_stats.logical_op_id
            full_op_id = record_op_stats.full_op_id
            if logical_op_id in self.operator_stats:
                if full_op_id in self.operator_stats[logical_op_id]:
                    self.operator_stats[logical_op_id][full_op_id] += record_op_stats
                else:
                    raise ValueError(f"RecordOpStats with full_op_id {full_op_id} not found in SentinelPlanStats")
            else:
                raise ValueError(f"RecordOpStats with logical_op_id {logical_op_id} not found in SentinelPlanStats")

    def __iadd__(self, plan_stats: SentinelPlanStats) -> None:
        """
        NOTE: we assume the execution layer guarantees:
        1. these plan_stats belong to the same plan
        2. these plan_stats come from sequential (non-overlapping) executions of the same plan

        The latter criteria implies it is okay for this method to sum the plan (and operator) runtimes.
        """
        self.total_plan_time += plan_stats.total_plan_time
        self.total_plan_cost += plan_stats.total_plan_cost
        for logical_op_id, physical_op_stats in plan_stats.operator_stats.items():
            for full_op_id, op_stats in physical_op_stats.items():
                if logical_op_id in self.operator_stats:
                    if full_op_id in self.operator_stats[logical_op_id]:
                        self.operator_stats[logical_op_id][full_op_id] += op_stats
                    else:
                        self.operator_stats[logical_op_id][full_op_id] = op_stats
                else:
                    self.operator_stats[logical_op_id] = physical_op_stats

    def __str__(self) -> str:
        stats = f"total_plan_time={self.total_plan_time} \n"
        stats += f"total_plan_cost={self.total_plan_cost} \n"
        for outer_idx, physical_op_stats in enumerate(self.operator_stats.values()):
            total_time = sum([op_stats.total_op_time for op_stats in physical_op_stats.values()])
            total_cost = sum([op_stats.total_op_cost for op_stats in physical_op_stats.values()])
            stats += f"{outer_idx}. total_time={total_time} total_cost={total_cost} \n"
            for inner_idx, op_stats in enumerate(physical_op_stats.values()):
                stats += f"    {outer_idx}.{inner_idx}. {op_stats.op_name} time={op_stats.total_op_time} cost={op_stats.total_op_cost} \n"
        return stats

    def to_json(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "plan_str": self.plan_str,
            "operator_stats": {
                logical_op_id: {full_op_id: op_stats.to_json() for full_op_id, op_stats in physical_op_stats.items()}
                for logical_op_id, physical_op_stats in self.operator_stats.items()
            },
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

    # dictionary of SentinelPlanStats objects (one for each sentinel plan run during execution)
    sentinel_plan_stats: dict[str, SentinelPlanStats] = field(default_factory=dict)

    # dictionary of PlanStats objects (one for each plan run during execution)
    plan_stats: dict[str, PlanStats] = field(default_factory=dict)

    # total time spent optimizing
    optimization_time: float = 0.0

    # total cost of optimizing
    optimization_cost: float = 0.0

    # total time spent executing the optimized plan
    plan_execution_time: float = 0.0

    # total cost of executing the optimized plan
    plan_execution_cost: float = 0.0

    # total runtime for the entire execution
    total_execution_time: float = 0.0

    # total cost for the entire execution
    total_execution_cost: float = 0.0

    # dictionary of sentinel plan strings; useful for printing executed sentinel plans in demos
    sentinel_plan_strs: dict[str, str] = field(default_factory=dict)

    # dictionary of plan strings; useful for printing executed plans in demos
    plan_strs: dict[str, str] = field(default_factory=dict)

    # start time for the execution; should be set by calling ExecutionStats.start()
    start_time: float | None = None

    # end time for the optimization; 
    optimization_end_time: float | None = None

    def start(self) -> None:
        """Start the timer for this execution."""
        self.start_time = time.time()

    def finish_optimization(self) -> None:
        """Finish the timer for the optimization phase of this execution."""
        if self.start_time is None:
            raise RuntimeError("ExecutionStats.start() must be called before ExecutionStats.finish_optimization()")

        # compute optimization time and cost
        self.optimization_end_time = time.time()
        self.optimization_time = self.optimization_end_time - self.start_time
        self.optimization_cost = self.sum_sentinel_plan_costs()

        # compute sentinel_plan_strs
        self.sentinel_plan_strs = {plan_id: plan_stats.plan_str for plan_id, plan_stats in self.sentinel_plan_stats.items()}

    def finish(self) -> None:
        """Finish the timer for this execution."""
        if self.start_time is None:
            raise RuntimeError("ExecutionStats.start() must be called before ExecutionStats.finish()")

        # compute time for plan and total execution
        end_time = time.time()
        self.plan_execution_time = (
            end_time - self.optimization_end_time
            if self.optimization_end_time is not None
            else end_time - self.start_time
        )
        self.total_execution_time = end_time - self.start_time

        # compute the cost for plan and total execution
        self.plan_execution_cost = self.sum_plan_costs()
        self.total_execution_cost = self.optimization_cost + self.plan_execution_cost

        # compute plan_strs
        self.plan_strs = {plan_id: plan_stats.plan_str for plan_id, plan_stats in self.plan_stats.items()}

    def sum_sentinel_plan_costs(self) -> float:
        """
        Sum the costs of all SentinelPlans in this execution.
        """
        return sum([plan_stats.sum_op_costs() for _, plan_stats in self.sentinel_plan_stats.items()])

    def sum_plan_costs(self) -> float:
        """
        Sum the costs of all PhysicalPlans in this execution.
        """
        return sum([plan_stats.sum_op_costs() for _, plan_stats in self.plan_stats.items()])

    def add_plan_stats(self, plan_stats: PlanStats | SentinelPlanStats | list[PlanStats] | list[SentinelPlanStats]) -> None:
        """
        Add the given PlanStats (or SentinelPlanStats) to this execution's plan stats.

        NOTE: we make the assumption that the same plan cannot be run more than once in parallel,
        i.e. each plan stats object for an individual plan comes from two different (sequential)
        periods in time. Thus, PlanStats objects can be summed.
        """
        # normalize input type to be list[PlanStats] or list[SentinelPlanStats]
        if isinstance(plan_stats, (PlanStats, SentinelPlanStats)):
            plan_stats = [plan_stats]

        for plan_stats_obj in plan_stats:
            if isinstance(plan_stats_obj, PlanStats) and plan_stats_obj.plan_id not in self.plan_stats:
                self.plan_stats[plan_stats_obj.plan_id] = plan_stats_obj
            elif isinstance(plan_stats_obj, PlanStats):
                self.plan_stats[plan_stats_obj.plan_id] += plan_stats_obj
            elif isinstance(plan_stats_obj, SentinelPlanStats) and plan_stats_obj.plan_id not in self.sentinel_plan_stats:
                self.sentinel_plan_stats[plan_stats_obj.plan_id] = plan_stats_obj
            elif isinstance(plan_stats_obj, SentinelPlanStats):
                self.sentinel_plan_stats[plan_stats_obj.plan_id] += plan_stats_obj
            else:
                raise TypeError(f"Cannot add {type(plan_stats)} to ExecutionStats")

    def clean_json(self, stats: dict):
        """
        Convert np.int64 and np.float64 to int and float for all values in stats.
        """
        for key, value in stats.items():
            if isinstance(value, dict):
                stats[key] = self.clean_json(value)
            elif isinstance(value, np.int64):
                stats[key] = int(value)
            elif isinstance(value, np.float64):
                stats[key] = float(value)
        return stats

    def to_json(self):
        stats = {
            "execution_id": self.execution_id,
            "sentinel_plan_stats": {
                plan_id: plan_stats.to_json() for plan_id, plan_stats in self.sentinel_plan_stats.items()
            },
            "plan_stats": {plan_id: plan_stats.to_json() for plan_id, plan_stats in self.plan_stats.items()},
            "optimization_time": self.optimization_time,
            "optimization_cost": self.optimization_cost,
            "plan_execution_time": self.plan_execution_time,
            "plan_execution_cost": self.plan_execution_cost,
            "total_execution_time": self.total_execution_time,
            "total_execution_cost": self.total_execution_cost,
            "sentinel_plan_strs": self.sentinel_plan_strs,
            "plan_strs": self.plan_strs,
        }
        stats = self.clean_json(stats)
        return stats


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

    def __rmul__(self, multiplier: float) -> OperatorCostEstimates:
        """
        Multiply all fields by a scalar.
        """
        dct = {field.name: getattr(self, field.name) * multiplier for field in fields(self)}
        return OperatorCostEstimates(**dct)

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
