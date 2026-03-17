from __future__ import annotations

import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from palimpzest.constants import (
    NAIVE_EST_NUM_GROUPS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    AggFunc,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Average, Count, Max, Min, Sum
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator

logger = logging.getLogger(__name__)


class AggregateOp(PhysicalOperator):
    """
    Aggregate operators accept a list of candidate DataRecords as input to their
    __call__ methods. Thus, we use a slightly modified abstract base class for
    these operators.
    """
    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        raise NotImplementedError("Using __call__ from abstract method")


class ApplyGroupByOp(AggregateOp):
    """
    Implementation of a GroupBy operator. This operator groups records by a set of fields
    and applies a function to each group.
    
    Can be initialized in two ways:
    1. Legacy: group_by_sig parameter containing fields and functions
    2. New: gby_fields, agg_fields, agg_funcs parameters directly
    """
    def __init__(self, gby_fields: list[str] = None, 
                 agg_fields: list[str] = None, agg_funcs: list[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # New API: construct group_by_sig from individual fields
        self.gby_fields = gby_fields
        self.agg_fields = agg_fields
        self.agg_funcs = agg_funcs

    def __str__(self):
        op = super().__str__()
        op += f"    Group-by Fields: {self.gby_fields}\n"
        op += f"    Agg. Fields: {self.agg_fields}\n"
        op += f"    Agg. Funcs: {self.agg_funcs}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "gby_fields": self.gby_fields, 
            "agg_fields": self.agg_fields, 
            "agg_funcs": self.agg_funcs,
            **id_params
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "gby_fields": self.gby_fields, 
            "agg_fields": self.agg_fields, 
            "agg_funcs": self.agg_funcs,
            **op_params
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the groupby takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=NAIVE_EST_NUM_GROUPS,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    @staticmethod
    def agg_init(func):
        if func.lower() == "count":
            return 0
        elif func.lower() == "average":
            return (0, 0)
        elif func.lower() == "sum":
            return 0
        elif func.lower() == "min":
            return float("inf")
        elif func.lower() == "max":
            return float("-inf")
        elif func.lower() == "list":
            return []
        elif func.lower() == "set":
            return set()
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_merge(func, state, val):
        if func.lower() == "count":
            return state + 1
        elif func.lower() == "average":
            sum_, cnt = state
            if val is None:
                return (sum_, cnt)
            return (sum_ + val, cnt + 1)
        elif func.lower() == "sum":
            if val is None:
                return state
            return state + sum(val) if isinstance(val, list) else state + val
        elif func.lower() == "min":
            if val is None:
                return state
            return min(state, min(val) if isinstance(val, list) else val)
        elif func.lower() == "max":
            if val is None:
                return state
            return max(state, max(val) if isinstance(val, list) else val)
        elif func.lower() == "list":
            state.append(val)
            return state
        elif func.lower() == "set":
            state.add(val)
            return state
        else:
            raise Exception("Unknown agg function " + func)

    @staticmethod
    def agg_final(func, state):
        if func.lower() in ["count", "sum", "min", "max", "list", "set"]:
            return state
        elif func.lower() == "average":
            sum, cnt = state
            return float(sum) / cnt if cnt > 0 else None
        else:
            raise Exception("Unknown agg function " + func)

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # build group array
        agg_state = {}
        for candidate in candidates:
            group = ()
            for f in self.gby_fields:
                if not hasattr(candidate, f):
                    raise TypeError(f"ApplyGroupByOp record missing expected field {f}")
                group = group + (getattr(candidate, f),)
            if group in agg_state:
                state = agg_state[group]
            else:
                state = []
                for fun in self.agg_funcs:
                    state.append(ApplyGroupByOp.agg_init(fun))
            for i in range(0, len(self.agg_funcs)):
                fun = self.agg_funcs[i]
                if not hasattr(candidate, self.agg_fields[i]):
                    raise TypeError(f"ApplyGroupByOp record missing expected field {self.agg_fields[i]}")
                field = getattr(candidate, self.agg_fields[i])
                state[i] = ApplyGroupByOp.agg_merge(fun, state[i], field)
            agg_state[group] = state

        # return list of data records (one per group)
        drs: list[DataRecord] = []
        group_by_fields = self.gby_fields
        for g in agg_state:
            # build up data item
            data_item = {}
            for i in range(0, len(g)):
                k = g[i]
                data_item[group_by_fields[i]] = k
            vals = agg_state[g]
            for i in range(0, len(vals)):
                v = ApplyGroupByOp.agg_final(self.agg_funcs[i], vals[i])
                data_item[self.agg_fields[i]] = v

            # create new DataRecord
            schema = self.output_schema
            data_item = schema(**data_item)
            dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)
            drs.append(dr)

        # create RecordOpStats objects
        total_time = time.time() - start_time
        record_op_stats_lst = []
        for dr in drs:
            record_op_stats = RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id,
                op_name=self.op_name(),
                time_per_record=total_time / len(drs),
                cost_per_record=0.0,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)

        # construct and return DataRecordSet
        return DataRecordSet(drs, record_op_stats_lst)

    def hierarchical_groupby(
        self,
        candidates: list[DataRecord],
        groupby_fields: list[list[str]],
        agg_fields: list[list[str]],
        agg_funcs: list[list[str]],
    ) -> dict:
        """
        Perform hierarchical (nested) exact groupby operations across multiple levels.

        At each intermediate level records are partitioned by exact field values without
        aggregation; the final level applies full aggregation via ApplyGroupByOp.__call__.

        Args:
            candidates: Input DataRecords.
            groupby_fields: List of lists of field names per level.
            agg_fields: List of lists of aggregate field names per level.
            agg_funcs: List of lists of aggregation function names per level.

        Returns:
            A DataRecordSet for a single level, or a nested dict for multiple levels.
        """
        from palimpzest.core.lib.schemas import create_groupby_schema_from_fields

        assert len(groupby_fields) == len(agg_fields) == len(agg_funcs), \
            "groupby_fields, agg_fields, and agg_funcs must all have the same length"

        def run_level(candidates, level):
            gby_names = groupby_fields[level]
            agg_names = agg_fields[level]
            funcs = agg_funcs[level]
            output_schema = create_groupby_schema_from_fields(gby_names, agg_names)
            op = ApplyGroupByOp(
                gby_fields=gby_names,
                agg_fields=agg_names,
                agg_funcs=funcs,
                output_schema=output_schema,
                input_schema=self.input_schema,
            )
            if level == len(groupby_fields) - 1:
                return op(candidates)
            outer_groups = {}
            for candidate in candidates:
                key = tuple(getattr(candidate, f, None) for f in gby_names)
                outer_groups.setdefault(key, []).append(candidate)
            return {key: run_level(grp, level + 1) for key, grp in outer_groups.items()}

        return run_level(candidates, 0)


class AverageAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        # enforce that output schema is correct
        assert kwargs["output_schema"].model_fields.keys() == Average.model_fields.keys(), "AverageAggregateOp requires output_schema to be Average"

        # enforce that input schema is a single numeric field
        input_field_types = list(kwargs["input_schema"].model_fields.values())
        assert len(input_field_types) == 1, "AverageAggregateOp requires input_schema to have exactly one field"
        numeric_field_types = [
            bool, int, float, int | float,
            bool | None, int | None, float | None, int | float | None,
            bool | Any, int | Any, float | Any, int | float | Any,
            bool | None | Any, int | None | Any, float | None | Any, int | float | None | Any,
        ]
        is_numeric = input_field_types[0].annotation in numeric_field_types
        assert is_numeric, f"AverageAggregateOp requires input_schema to have a numeric field type, i.e. one of: {numeric_field_types}\nGot: {input_field_types[0]}"

        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # NOTE: we currently do not guarantee that input values conform to their specified type;
        #       as a result, we simply omit any values which do not parse to a float from the average
        # NOTE: right now we perform a check in the constructor which enforces that the input_schema
        #       has a single field which is numeric in nature; in the future we may want to have a
        #       cleaner way of computing the value (rather than `float(list(candidate...))` below)
        summation, total = 0, 0
        for candidate in candidates:
            try:
                summation += float(list(candidate.to_dict().values())[0])
                total += 1
            except Exception:
                pass
        data_item = Average(average=summation / total)
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return DataRecordSet([dr], [record_op_stats])


class SumAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        # enforce that output schema is correct
        assert kwargs["output_schema"].model_fields.keys() == Sum.model_fields.keys(), "SumAggregateOp requires output_schema to be Sum"

        # enforce that input schema is a single numeric field
        input_field_types = list(kwargs["input_schema"].model_fields.values())
        assert len(input_field_types) == 1, "SumAggregateOp requires input_schema to have exactly one field"
        numeric_field_types = [
            bool, int, float, int | float,
            bool | None, int | None, float | None, int | float | None,
            bool | Any, int | Any, float | Any, int | float | Any,
            bool | None | Any, int | None | Any, float | None | Any, int | float | None | Any,
        ]
        is_numeric = input_field_types[0].annotation in numeric_field_types
        assert is_numeric, f"SumAggregateOp requires input_schema to have a numeric field type, i.e. one of: {numeric_field_types}\nGot: {input_field_types[0]}"

        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # NOTE: we currently do not guarantee that input values conform to their specified type;
        #       as a result, we simply omit any values which do not parse to a float from the average
        # NOTE: right now we perform a check in the constructor which enforces that the input_schema
        #       has a single field which is numeric in nature; in the future we may want to have a
        #       cleaner way of computing the value (rather than `float(list(candidate...))` below)
        summation = 0
        for candidate in candidates:
            with contextlib.suppress(Exception):
                summation += float(list(candidate.to_dict().values())[0])
        data_item = Sum(sum=summation)
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
        )

        return DataRecordSet([dr], [record_op_stats])


class CountAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        # enforce that output schema is correct
        assert kwargs["output_schema"].model_fields.keys() == Count.model_fields.keys(), "CountAggregateOp requires output_schema to be Count"

        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # create new DataRecord
        data_item = Count(count=len(candidates))
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])


class MinAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        # enforce that output schema is correct
        assert kwargs["output_schema"].model_fields.keys() == Min.model_fields.keys(), "MinAggregateOp requires output_schema to be Min"

        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # create new DataRecord
        min = float("inf")
        for candidate in candidates:
            try:  # noqa: SIM105
                min = min(float(list(candidate.to_dict().values())[0]), min)
            except Exception:
                pass
        data_item = Min(min=min if min != float("inf") else None)
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr.id,
            record_parent_ids=dr.parent_ids,
            record_source_indices=dr.source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])


class MaxAggregateOp(AggregateOp):
    # NOTE: we don't actually need / use agg_func here (yet)

    def __init__(self, agg_func: AggFunc, *args, **kwargs):
        # enforce that output schema is correct
        assert kwargs["output_schema"].model_fields.keys() == Max.model_fields.keys(), "MaxAggregateOp requires output_schema to be Max"

        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    def __str__(self):
        op = super().__str__()
        op += f"    Function: {str(self.agg_func)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {"agg_func": str(self.agg_func), **id_params}

    def get_op_params(self):
        op_params = super().get_op_params()
        return {"agg_func": self.agg_func, **op_params}

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        # for now, assume applying the aggregation takes negligible additional time (and no cost in USD)
        return OperatorCostEstimates(
            cardinality=1,
            time_per_record=0,
            cost_per_record=0,
            quality=1.0,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        # create new DataRecord
        
        max = float("-inf")
        for candidate in candidates:
            try:  # noqa: SIM105
                max = max(float(list(candidate.to_dict().values())[0]), max)
            except Exception:
                pass
        data_item = Max(max=max if max != float("-inf") else None)
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr.id,
            record_parent_ids=dr.parent_ids,
            record_source_indices=dr.source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=0.0,
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])


class SemanticAggregate(AggregateOp):

    def __init__(self, agg_str: str, model: Model, prompt_strategy: PromptStrategy = PromptStrategy.AGG, reasoning_effort: str = "default", *args, **kwargs):
        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_str = agg_str
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        if model is not None:
            self.generator = Generator(model, prompt_strategy, reasoning_effort)

    def __str__(self):
        op = super().__str__()
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        op += f"    Reasoning Effort: {self.reasoning_effort}\n"
        op += f"    Agg: {str(self.agg_str)}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        id_params = {
            "agg_str": self.agg_str,
            "model": None if self.model is None else self.model.value,
            "prompt_strategy": None if self.prompt_strategy is None else self.prompt_strategy.value,
            "reasoning_effort": self.reasoning_effort,
            **id_params,
        }

        return id_params

    def get_op_params(self):
        op_params = super().get_op_params()
        op_params = {
            "agg_str": self.agg_str,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            **op_params,
        }

        return op_params

    def get_model_name(self) -> str:
        return self.model.value

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the LLMConvert operation. Implicitly, these estimates
        assume the use of a single LLM call for each input record. Child classes of LLMConvert
        may call this function through super() and adjust these estimates as needed (or they can
        completely override this function).
        """
        # estimate number of input and output tokens from source
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS * source_op_cost_estimates.cardinality
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS

        # get est. of conversion time per record from model card;
        model_conversion_time_per_record = self.model.get_seconds_per_output_token() * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        usd_per_input_token = self.model.get_usd_per_input_token()
        if getattr(self, "prompt_strategy", None) is not None and self.is_audio_op():
            usd_per_input_token = self.model.get_usd_per_audio_input_token()

        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + self.model.get_usd_per_output_token() * est_num_output_tokens
        )

        # estimate quality of output based on the strength of the model being used
        quality = self.model.get_overall_score() / 100.0

        return OperatorCostEstimates(
            cardinality=1.0,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        start_time = time.time()

        if len(candidates) == 0:
            return DataRecordSet([], [])

        # get the set of input fields to use for the operation
        input_fields = self.get_input_fields()

        # get the set of output fields to use for the operation
        fields_to_generate = self.get_fields_to_generate(candidates[0])
        fields = {field: field_type for field, field_type in self.output_schema.model_fields.items() if field in fields_to_generate}

        # construct kwargs for generation
        gen_kwargs = {"project_cols": input_fields, "output_schema": self.output_schema, "agg_instruction": self.agg_str}

        # generate outputs for all fields in a single query
        field_answers, _, generation_stats, _ = self.generator(candidates, fields, **gen_kwargs)
        assert all([field in field_answers for field in fields]), "Not all fields were generated!"

        # construct data record for the output
        field, value = fields_to_generate[0], field_answers[fields_to_generate[0]][0]
        data_item = self.output_schema(**{field: value})
        dr = DataRecord.from_agg_parents(data_item, parent_records=candidates)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=time.time() - start_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=self.get_model_name(),
            answer={field: value},
            input_fields=input_fields,
            generated_fields=fields_to_generate,
            input_text_tokens=generation_stats.input_text_tokens,
            input_audio_tokens=generation_stats.input_audio_tokens,
            input_image_tokens=generation_stats.input_image_tokens,
            cache_read_tokens=generation_stats.cache_read_tokens,
            cache_creation_tokens=generation_stats.cache_creation_tokens,
            output_text_tokens=generation_stats.output_text_tokens,
            embedding_input_tokens=generation_stats.embedding_input_tokens,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            image_operation=self.is_image_op(),
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])


# ---------------------------------------------------------------------------
# Constants for batching / parallelism defaults
# ---------------------------------------------------------------------------
DEFAULT_GROUPBY_BATCH_SIZE = 10
"""Default number of records to send in a single LLM call for group assignment."""

DEFAULT_GROUPBY_PARALLELISM = 8
"""Default number of concurrent threads for LLM calls in semantic groupby."""

DEFAULT_AGG_PARALLELISM = 4
"""Default number of concurrent threads for semantic aggregation across groups."""

# Standard (non-semantic) aggregation function names recognised by the operator.
STANDARD_AGG_FUNCS = frozenset({"avg", "average", "count", "sum", "min", "max", "list", "set"})


class SemanticGroupByOp(AggregateOp):
    """Semantic GroupBy operator backed by LLM calls.

    This operator supports:
    * **Semantic grouping** -- the LLM determines which group each record belongs
      to based on a natural-language description.
    * **Exact grouping** -- records are partitioned by literal field values (no LLM
      needed for the grouping phase).
    * **Standard aggregation** -- count / sum / avg / min / max / list / set applied
      per-group without an LLM.
    * **Semantic aggregation** -- an LLM-based aggregation function (e.g. "summarise
      the most positive review") applied per-group.

    Optimisation knobs
    ------------------
    ``batch_size``
        Number of records to include in a *single* LLM prompt when assigning
        groups (Phase 1).  Larger batches amortise prompt overhead but increase
        context length and risk of the model losing track of records.  Set to 1
        to fall back to one-record-at-a-time mode.

    ``groupby_parallelism``
        Number of concurrent ``ThreadPoolExecutor`` workers for the LLM calls in
        the grouping phase.  Each worker processes one batch.  This is modelled
        after ``join_parallelism`` in ``NestedLoopsJoin``.

    ``agg_parallelism``
        Number of concurrent workers for semantic aggregation calls (one call per
        group x semantic-agg-field combination).
    """

    def __init__(
        self,
        gby_fields: list[str] | list[dict],
        agg_fields: list[str] | list[dict],
        agg_funcs: list[str],
        model: Model | None = None,
        prompt_strategy: PromptStrategy = PromptStrategy.AGG,
        reasoning_effort: str | None = None,
        batch_size: int = DEFAULT_GROUPBY_BATCH_SIZE,
        groupby_parallelism: int = DEFAULT_GROUPBY_PARALLELISM,
        agg_parallelism: int = DEFAULT_AGG_PARALLELISM,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # -- field specs -------------------------------------------------
        self.gby_fields_spec = gby_fields
        self.agg_fields_spec = agg_fields

        # Extract plain field names for backward compatibility / quick access
        self.gby_fields = [f["name"] if isinstance(f, dict) else f for f in gby_fields]
        self.agg_fields = [f["name"] if isinstance(f, dict) else f for f in agg_fields]

        self.agg_funcs = agg_funcs
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort

        # -- optimisation knobs ------------------------------------------
        self.batch_size = max(1, batch_size)
        self.groupby_parallelism = max(1, groupby_parallelism)
        self.agg_parallelism = max(1, agg_parallelism)

        # -- generator (lazily initialised for exact-only operators) -----
        self._generator: Generator | None = None
        if self.model is not None:
            self._generator = Generator(
                self.model,
                self.prompt_strategy,
                self.reasoning_effort,
            )

        # Thread-safety lock for stats accumulation
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------
    @property
    def generator(self) -> Generator:
        """Return the generator, raising if not initialised."""
        if self._generator is None:
            raise RuntimeError(
                "SemanticGroupByOp.generator accessed but no model was provided. "
                "Semantic operations require a model."
            )
        return self._generator

    def get_model_name(self) -> str | None:
        return self.model.value if self.model is not None else None

    # ------------------------------------------------------------------
    # Repr helpers
    # ------------------------------------------------------------------
    def __str__(self):
        op = super().__str__()
        op += f"    Group-by Fields: {self.gby_fields}\n"
        op += f"    Agg. Fields: {self.agg_fields}\n"
        op += f"    Agg. Funcs: {self.agg_funcs}\n"
        if self.model is not None:
            op += f"    Model: {self.model.value}\n"
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        op += f"    Batch Size: {self.batch_size}\n"
        op += f"    GroupBy Parallelism: {self.groupby_parallelism}\n"
        op += f"    Agg Parallelism: {self.agg_parallelism}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "gby_fields": self.gby_fields,
            "agg_fields": self.agg_fields,
            "agg_funcs": self.agg_funcs,
            "model": self.model.value if self.model else None,
            "prompt_strategy": self.prompt_strategy.value if self.prompt_strategy else None,
            "reasoning_effort": self.reasoning_effort,
            "batch_size": self.batch_size,
            **id_params,
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "gby_fields": self.gby_fields_spec,
            "agg_fields": self.agg_fields_spec,
            "agg_funcs": self.agg_funcs,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            "batch_size": self.batch_size,
            "groupby_parallelism": self.groupby_parallelism,
            "agg_parallelism": self.agg_parallelism,
            **op_params,
        }

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------
    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """Naive cost estimate -- follows the same pattern as ``SemanticAggregate``."""
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS * source_op_cost_estimates.cardinality
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS * NAIVE_EST_NUM_GROUPS

        if self.model is None:
            # Exact-only groupby: negligible cost
            return OperatorCostEstimates(
                cardinality=NAIVE_EST_NUM_GROUPS,
                time_per_record=0,
                cost_per_record=0,
                quality=1.0,
            )

        time_per_record = self.model.get_seconds_per_output_token() * est_num_output_tokens

        usd_per_input_token = self.model.get_usd_per_input_token()
        if getattr(self, "prompt_strategy", None) is not None and self.is_audio_op():
            usd_per_input_token = self.model.get_usd_per_audio_input_token()

        cost_per_record = (
            usd_per_input_token * est_num_input_tokens
            + self.model.get_usd_per_output_token() * est_num_output_tokens
        )

        quality = self.model.get_overall_score() / 100.0

        return OperatorCostEstimates(
            cardinality=NAIVE_EST_NUM_GROUPS,
            time_per_record=time_per_record,
            cost_per_record=cost_per_record,
            quality=quality,
        )

    # ==================================================================
    #  MAIN ENTRY POINT
    # ==================================================================
    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        """Execute the semantic group-by operation.

        The pipeline has three phases:

        1. **Grouping** -- assign each record to a group key (semantic or exact).
        2. **Partitioning** -- bucket records by their group key.
        3. **Aggregation** -- compute each agg function per group (semantic or
           standard).

        Batching and parallelism are applied in Phase 1 and Phase 3.
        """
        start_time = time.time()

        if len(candidates) == 0:
            return DataRecordSet([], [])

        # Detect modes
        # A field is semantic if it was user-provided as a dict (needs LLM inference).
        # Fields derived from plain column names have 'semantic': False.
        is_semantic_gby = any(
            (isinstance(f, dict) and f.get('semantic', True))
            for f in self.gby_fields_spec
        )
        is_semantic_agg = any(f.lower() not in STANDARD_AGG_FUNCS for f in self.agg_funcs)

        # Phase 1 -- grouping
        group_assignments, groupby_stats = self._perform_groupby(candidates, is_semantic_gby)

        # Phase 2 -- partition
        grouped_records = self._partition_by_group(candidates, group_assignments)

        # Phase 3 -- aggregation
        drs, stats_lst = self._perform_aggregation(
            grouped_records, is_semantic_agg, groupby_stats, candidates, start_time,
        )

        return DataRecordSet(drs, stats_lst)

    # ==================================================================
    #  PHASE 1: GROUPING
    # ==================================================================
    def _perform_groupby(
        self,
        candidates: list[DataRecord],
        is_semantic: bool,
    ) -> tuple[list[tuple], GenerationStats]:
        """Route to semantic or exact grouping."""
        if is_semantic:
            return self._perform_semantic_groupby(candidates)
        return self._perform_exact_groupby(candidates)

    # -- exact groupby -------------------------------------------------
    def _perform_exact_groupby(
        self,
        candidates: list[DataRecord],
    ) -> tuple[list[tuple], GenerationStats]:
        """Group records by literal field values -- no LLM needed."""
        assignments: list[tuple] = []
        for candidate in candidates:
            key = tuple(getattr(candidate, f, None) for f in self.gby_fields)
            assignments.append(key)
        return assignments, GenerationStats()

    # -- semantic groupby (batched + parallel) -------------------------
    def _perform_semantic_groupby(
        self,
        candidates: list[DataRecord],
    ) -> tuple[list[tuple], GenerationStats]:
        """Assign records to groups via LLM, with batching & parallelism.

        Records are split into batches of ``self.batch_size`` and submitted
        to a ``ThreadPoolExecutor`` with ``self.groupby_parallelism`` workers.
        """
        from palimpzest.core.lib.schemas import create_schema_from_fields

        # Build a tiny schema that the LLM fills in for each record
        gby_schema_fields = []
        for spec in self.gby_fields_spec:
            if isinstance(spec, dict):
                gby_schema_fields.append({
                    "name": spec["name"],
                    "type": spec.get("type", str),
                    "desc": spec.get("desc", f"Semantic group for {spec['name']}"),
                })
            else:
                gby_schema_fields.append({
                    "name": spec,
                    "type": str,
                    "desc": f"The semantic category for {spec}",
                })
        groupby_schema = create_schema_from_fields(gby_schema_fields)

        # Natural-language instruction for the LLM
        field_descs = "; ".join(
            f"'{s['name']}': {s.get('desc', s['name'])}"
            for s in (
                self.gby_fields_spec
                if all(isinstance(s, dict) for s in self.gby_fields_spec)
                else gby_schema_fields
            )
        )
        agg_instruction = (
            f"Categorise each input record into a semantic group. "
            f"The grouping fields and their descriptions are: {field_descs}. "
            f"Return the group label(s) for each record."
        )

        # Split candidates into batches
        batches: list[list[DataRecord]] = [
            candidates[i : i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]

        # Prepare output containers (order-preserving)
        all_labels: list[list[str | tuple] | None] = [None] * len(batches)
        accumulated_stats = GenerationStats()

        logger.info(
            "SemanticGroupByOp: assigning %d records across %d batches "
            "(batch_size=%d, parallelism=%d)",
            len(candidates), len(batches), self.batch_size, self.groupby_parallelism,
        )

        def _process_batch(
            batch_idx: int, batch: list[DataRecord],
        ) -> tuple[int, list[str | tuple], GenerationStats]:
            """Process a single batch of records through the LLM."""
            batch_labels: list[str | tuple] = []
            batch_stats = GenerationStats()

            input_fields = list(self.gby_fields)
            fields = {f: str for f in self.gby_fields}

            gen_kwargs = {
                "project_cols": input_fields,
                "output_schema": groupby_schema,
                "agg_instruction": agg_instruction,
            }

            if len(batch) == 1:
                # Single-record batch -- call generator directly
                field_answers, _, gen_stats, _ = self.generator(
                    batch[0], fields, **gen_kwargs,
                )
                label = self._extract_group_label(field_answers)
                batch_labels.append(label)
                if gen_stats is not None:
                    batch_stats += gen_stats
            else:
                # Multi-record batch -- pass list of candidates
                field_answers, _, gen_stats, _ = self.generator(
                    batch, fields, **gen_kwargs,
                )
                if gen_stats is not None:
                    batch_stats += gen_stats

                # The generator may return a list per field or a single value
                # depending on cardinality; normalise to one label per record
                batch_labels = self._extract_batch_group_labels(
                    field_answers, len(batch),
                )

            return batch_idx, batch_labels, batch_stats

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=self.groupby_parallelism) as executor:
            futures = {
                executor.submit(_process_batch, idx, batch): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx, labels, stats = future.result()
                all_labels[batch_idx] = labels
                with self._stats_lock:
                    accumulated_stats += stats

        # Flatten ordered labels -> one tuple per candidate
        group_assignments: list[tuple] = []
        for batch_labels in all_labels:
            for label in batch_labels:
                if isinstance(label, tuple):
                    group_assignments.append(label)
                else:
                    group_assignments.append((label,))

        logger.info(
            "SemanticGroupByOp: found %d unique groups from %d records",
            len(set(group_assignments)), len(candidates),
        )

        return group_assignments, accumulated_stats

    # -- label extraction helpers --------------------------------------
    @staticmethod
    def _coerce_to_str(val) -> str:
        """Unwrap nested lists and coerce to a hashable string."""
        while isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if val is None:
            return "unknown"
        return str(val)

    def _extract_group_label(self, field_answers: dict) -> str | tuple:
        """Extract a single group label from generator output."""
        if len(self.gby_fields) == 1:
            val = field_answers.get(self.gby_fields[0])
            return self._coerce_to_str(val)

        # Multi-column groupby -> tuple
        parts = []
        for f in self.gby_fields:
            val = field_answers.get(f)
            parts.append(self._coerce_to_str(val))
        return tuple(parts)

    @staticmethod
    def _unwrap_generator_list(vals: list) -> list:
        """Unwrap the extra nesting added by Generator._prepare_field_answers.

        The Generator with ONE_TO_ONE cardinality wraps every field value in a
        list, so ``["a", "b", "c"]`` becomes ``[["a", "b", "c"]]``.  For
        batch group-label extraction we need the inner flat list.
        """
        if len(vals) == 1 and isinstance(vals[0], list):
            return vals[0]
        return vals

    def _extract_batch_group_labels(
        self, field_answers: dict, batch_size: int,
    ) -> list[str | tuple]:
        """Extract per-record group labels from a batched generator response."""
        labels: list[str | tuple] = []

        if len(self.gby_fields) == 1:
            field = self.gby_fields[0]
            vals = field_answers.get(field, [])
            if not isinstance(vals, list):
                vals = [vals]

            # Unwrap double-nesting from Generator._prepare_field_answers
            vals = self._unwrap_generator_list(vals)

            # Pad / truncate to batch_size
            while len(vals) < batch_size:
                vals.append("unknown")
            for v in vals[:batch_size]:
                labels.append(self._coerce_to_str(v))
        else:
            # Multi-column: zip columns together
            columns = []
            for f in self.gby_fields:
                col_vals = field_answers.get(f, [])
                if not isinstance(col_vals, list):
                    col_vals = [col_vals]

                # Unwrap double-nesting from Generator._prepare_field_answers
                col_vals = self._unwrap_generator_list(col_vals)

                while len(col_vals) < batch_size:
                    col_vals.append("unknown")
                columns.append(col_vals[:batch_size])

            for row_vals in zip(*columns):
                labels.append(
                    tuple(self._coerce_to_str(v) for v in row_vals),
                )

        return labels

    # ==================================================================
    #  PHASE 2: PARTITION
    # ==================================================================
    @staticmethod
    def _partition_by_group(
        candidates: list[DataRecord],
        group_assignments: list[tuple],
    ) -> dict[tuple, list[DataRecord]]:
        """Bucket candidates into a dict keyed by their group assignment."""
        grouped: dict[tuple, list[DataRecord]] = {}
        for candidate, key in zip(candidates, group_assignments):
            grouped.setdefault(key, []).append(candidate)
        return grouped

    # ==================================================================
    #  PHASE 3: AGGREGATION
    # ==================================================================
    def _perform_aggregation(
        self,
        grouped_records: dict[tuple, list[DataRecord]],
        is_semantic_agg: bool,
        groupby_stats: GenerationStats,
        all_candidates: list[DataRecord],
        start_time: float,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        """Dispatch to exact or semantic aggregation."""
        if is_semantic_agg:
            return self._aggregate_semantic(
                grouped_records, groupby_stats, all_candidates, start_time,
            )
        return self._aggregate_exact(
            grouped_records, groupby_stats, all_candidates, start_time,
        )

    # -- exact aggregation ---------------------------------------------
    def _aggregate_exact(
        self,
        grouped_records: dict[tuple, list[DataRecord]],
        groupby_stats: GenerationStats,
        all_candidates: list[DataRecord],
        start_time: float,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        """Apply standard agg functions (count/sum/...) per group -- no LLM."""
        drs: list[DataRecord] = []
        stats_lst: list[RecordOpStats] = []
        output_field_names = [
            f for f in self.output_schema.model_fields if f not in self.gby_fields
        ]
        num_groups = len(grouped_records)

        for group_key, group_candidates in grouped_records.items():
            # Initialise & merge aggregation state
            state = [ApplyGroupByOp.agg_init(fun) for fun in self.agg_funcs]
            for candidate in group_candidates:
                for i, (fun, agg_field) in enumerate(
                    zip(self.agg_funcs, self.agg_fields),
                ):
                    if not hasattr(candidate, agg_field):
                        raise TypeError(
                            f"SemanticGroupByOp record missing expected field {agg_field}"
                        )
                    state[i] = ApplyGroupByOp.agg_merge(
                        fun, state[i], getattr(candidate, agg_field),
                    )

            # Build output data item
            data_item: dict[str, Any] = {}
            for i, gby_field in enumerate(self.gby_fields):
                data_item[gby_field] = group_key[i]
            for i, agg_func in enumerate(self.agg_funcs):
                data_item[output_field_names[i]] = ApplyGroupByOp.agg_final(
                    agg_func, state[i],
                )

            dr = DataRecord.from_agg_parents(
                self.output_schema(**data_item), parent_records=all_candidates,
            )
            drs.append(dr)

            cost = (
                groupby_stats.cost_per_record / num_groups
                if groupby_stats.cost_per_record > 0
                else 0.0
            )
            stats_lst.append(
                RecordOpStats(
                    record_id=dr._id,
                    record_parent_ids=dr._parent_ids,
                    record_source_indices=dr._source_indices,
                    record_state=dr.to_dict(include_bytes=False),
                    full_op_id=self.get_full_op_id(),
                    logical_op_id=self.logical_op_id or "semantic-groupby",
                    op_name=self.op_name(),
                    time_per_record=(time.time() - start_time) / num_groups,
                    cost_per_record=cost,
                    model_name=self.get_model_name(),
                    input_fields=self.get_input_fields(),
                    generated_fields=list(self.output_schema.model_fields.keys()),
                    input_text_tokens=groupby_stats.input_text_tokens / num_groups,
                    output_text_tokens=groupby_stats.output_text_tokens / num_groups,
                    llm_call_duration_secs=groupby_stats.llm_call_duration_secs / num_groups,
                    total_llm_calls=groupby_stats.total_llm_calls / num_groups,
                    op_details={k: str(v) for k, v in self.get_id_params().items()},
                )
            )

        return drs, stats_lst

    # -- semantic aggregation (parallel across groups) -----------------
    def _aggregate_semantic(
        self,
        grouped_records: dict[tuple, list[DataRecord]],
        groupby_stats: GenerationStats,
        all_candidates: list[DataRecord],
        start_time: float,
    ) -> tuple[list[DataRecord], list[RecordOpStats]]:
        """Apply aggregation per group; semantic agg functions use the LLM.

        Groups are processed in parallel with ``self.agg_parallelism`` workers.
        """
        num_groups = len(grouped_records)
        output_field_names = [
            f for f in self.output_schema.model_fields if f not in self.gby_fields
        ]

        # Container for ordered results
        ordered_keys = list(grouped_records.keys())
        results: list[tuple[DataRecord, RecordOpStats] | None] = [None] * num_groups

        def _aggregate_one_group(
            idx: int, group_key: tuple,
        ) -> tuple[int, DataRecord, RecordOpStats]:
            """Aggregate a single group (may involve LLM calls)."""
            group_candidates = grouped_records[group_key]
            data_item: dict[str, Any] = {}
            group_agg_stats = GenerationStats()

            # Group-by field values
            for i, gby_field in enumerate(self.gby_fields):
                data_item[gby_field] = group_key[i]

            # Aggregate each field
            for i, (agg_func, agg_field) in enumerate(
                zip(self.agg_funcs, self.agg_fields),
            ):
                if agg_func.lower() not in STANDARD_AGG_FUNCS:
                    # Semantic aggregation via LLM
                    value, gen_stats = self._apply_semantic_agg_llm(
                        group_candidates, agg_field, agg_func,
                    )
                    group_agg_stats += gen_stats
                else:
                    # Standard aggregation
                    state = ApplyGroupByOp.agg_init(agg_func)
                    for candidate in group_candidates:
                        if not hasattr(candidate, agg_field):
                            raise TypeError(
                                f"SemanticGroupByOp record missing expected field "
                                f"{agg_field}"
                            )
                        state = ApplyGroupByOp.agg_merge(
                            agg_func, state, getattr(candidate, agg_field),
                        )
                    value = ApplyGroupByOp.agg_final(agg_func, state)

                data_item[output_field_names[i]] = value

            dr = DataRecord.from_agg_parents(
                self.output_schema(**data_item), parent_records=all_candidates,
            )

            combined = groupby_stats + group_agg_stats
            record_op_stats = RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id or "semantic-groupby",
                op_name=self.op_name(),
                time_per_record=(time.time() - start_time) / num_groups,
                cost_per_record=combined.cost_per_record / num_groups,
                model_name=self.get_model_name(),
                input_fields=self.get_input_fields(),
                generated_fields=list(self.output_schema.model_fields.keys()),
                input_text_tokens=combined.input_text_tokens / num_groups,
                output_text_tokens=combined.output_text_tokens / num_groups,
                llm_call_duration_secs=combined.llm_call_duration_secs / num_groups,
                total_llm_calls=combined.total_llm_calls / num_groups,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )

            return idx, dr, record_op_stats

        # Execute group aggregations in parallel
        with ThreadPoolExecutor(max_workers=self.agg_parallelism) as executor:
            futures = {
                executor.submit(_aggregate_one_group, idx, key): idx
                for idx, key in enumerate(ordered_keys)
            }
            for future in as_completed(futures):
                idx, dr, stats = future.result()
                results[idx] = (dr, stats)

        drs = [r[0] for r in results]  # type: ignore[index]
        stats_lst = [r[1] for r in results]  # type: ignore[index]
        return drs, stats_lst

    # -- single semantic aggregation call ------------------------------
    def _apply_semantic_agg_llm(
        self,
        group_candidates: list[DataRecord],
        agg_field: str,
        agg_func: str,
    ) -> tuple[Any, GenerationStats]:
        """Call the LLM to perform a semantic aggregation on *group_candidates*.

        Args:
            group_candidates: Records belonging to one group.
            agg_field: The field name being aggregated.
            agg_func: Natural-language description of the aggregation
                      (e.g. ``"most positive review"``).

        Returns:
            ``(aggregated_value, generation_stats)``
        """
        from palimpzest.core.lib.schemas import create_schema_from_fields

        # Determine output type for this field
        field_type: type = str
        for spec in self.agg_fields_spec:
            if isinstance(spec, dict) and spec.get("name") == agg_field:
                field_type = spec.get("type", str)
                break
        else:
            if agg_field in self.output_schema.model_fields:
                field_type = self.output_schema.model_fields[agg_field].annotation or str

        agg_schema = create_schema_from_fields([
            {"name": agg_field, "type": field_type, "desc": agg_func},
        ])

        agg_instruction = (
            f"Apply the following aggregation: {agg_func} on field '{agg_field}'"
        )
        input_fields = [agg_field]
        fields = {agg_field: field_type}

        gen_kwargs = {
            "project_cols": input_fields,
            "output_schema": agg_schema,
            "agg_instruction": agg_instruction,
        }

        field_answers, _, gen_stats, _ = self.generator(
            group_candidates, fields, **gen_kwargs,
        )

        value = None
        answer = field_answers.get(agg_field)
        if isinstance(answer, list) and len(answer) > 0:
            value = answer[0]
        elif answer is not None:
            value = answer

        return value, gen_stats if gen_stats is not None else GenerationStats()

    # ==================================================================
    #  HIERARCHICAL GROUPBY
    # ==================================================================
    def hierarchical_groupby(
        self,
        candidates: list[DataRecord],
        groupby_fields: list[list[str | dict]],
        agg_fields: list[list[str | dict]],
        agg_funcs: list[list[str]],
        model: Model | None = None,
        prompt_strategy: PromptStrategy = PromptStrategy.AGG,
        reasoning_effort: str | None = None,
    ) -> dict:
        """Perform hierarchical (nested) semantic groupby operations.

        At each intermediate level the LLM assigns group labels to the original
        records (without aggregation) so that inner levels operate on the same
        raw records.  The final level runs a full semantic groupby with
        aggregation.

        Args:
            candidates: Input DataRecords.
            groupby_fields: List of lists of field specs per level.
            agg_fields: List of lists of aggregate field specs per level.
            agg_funcs: List of lists of aggregation function names per level.
            model: Optional LLM model override (falls back to ``self.model``).
            prompt_strategy: Prompt strategy (defaults to AGG).
            reasoning_effort: Optional reasoning effort override.

        Returns:
            A ``DataRecordSet`` for a single level, or a nested dict for
            multiple levels.
        """
        from palimpzest.core.lib.schemas import create_groupby_schema_from_fields

        assert len(groupby_fields) == len(agg_fields) == len(agg_funcs), (
            "groupby_fields, agg_fields, and agg_funcs must all have the same length"
        )

        def _normalize(fields):
            return [
                f
                if isinstance(f, dict)
                else {"name": f, "desc": f"Group by {f}", "type": str}
                for f in fields
            ]

        _model = model or self.model
        _ps = prompt_strategy or self.prompt_strategy
        _re = reasoning_effort or self.reasoning_effort

        def _run_level(cands, level):
            gby_specs = _normalize(groupby_fields[level])
            agg_specs = _normalize(agg_fields[level])
            funcs = agg_funcs[level]
            gby_names = [s["name"] for s in gby_specs]
            agg_names = [s["name"] for s in agg_specs]
            out_schema = create_groupby_schema_from_fields(gby_names, agg_names)

            op = SemanticGroupByOp(
                gby_fields=gby_specs,
                agg_fields=agg_specs,
                agg_funcs=funcs,
                model=_model,
                prompt_strategy=_ps,
                reasoning_effort=_re,
                batch_size=self.batch_size,
                groupby_parallelism=self.groupby_parallelism,
                agg_parallelism=self.agg_parallelism,
                output_schema=out_schema,
                input_schema=self.input_schema,
            )

            if level == len(groupby_fields) - 1:
                return op(cands)

            # Intermediate: assign labels, forward raw records
            labels, _ = op._perform_semantic_groupby(cands)
            outer_groups: dict[tuple, list[DataRecord]] = {}
            for cand, label in zip(cands, labels):
                key = label if isinstance(label, tuple) else (label,)
                outer_groups.setdefault(key, []).append(cand)
            return {
                key: _run_level(grp, level + 1)
                for key, grp in outer_groups.items()
            }

        return _run_level(candidates, 0)
