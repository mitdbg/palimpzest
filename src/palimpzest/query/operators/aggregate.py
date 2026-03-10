from __future__ import annotations

import contextlib
import time
from typing import Any

from palimpzest.constants import (
    MODEL_CARDS,
    NAIVE_EST_NUM_GROUPS,
    NAIVE_EST_NUM_INPUT_TOKENS,
    NAIVE_EST_NUM_OUTPUT_TOKENS,
    AggFunc,
    Model,
    PromptStrategy,
)
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.lib.schemas import Average, Count, Max, Min, Sum
from palimpzest.core.models import OperatorCostEstimates, RecordOpStats, GenerationStats
from palimpzest.query.generators.generators import Generator
from palimpzest.query.operators.physical import PhysicalOperator


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
        # Construct aggregation field names: "func(field)"
        agg_field_names = [f"{field}" for field in self.agg_fields]
        for g in agg_state:
            # build up data item
            data_item = {}
            for i in range(0, len(g)):
                k = g[i]
                data_item[group_by_fields[i]] = k
            vals = agg_state[g]
            for i in range(0, len(vals)):
                v = ApplyGroupByOp.agg_final(self.agg_funcs[i], vals[i])
                data_item[agg_field_names[i]] = v

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

    def __init__(self, agg_str: str, model: Model, prompt_strategy: PromptStrategy = PromptStrategy.AGG, reasoning_effort: str | None = None, *args, **kwargs):
        # call parent constructor
        super().__init__(*args, **kwargs)
        self.agg_str = agg_str
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        if model is not None:
            self.generator = Generator(model, prompt_strategy, reasoning_effort, self.api_base)

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
        model_name = self.model.value
        model_conversion_time_per_record = MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        # Check for audio models first
        if "usd_per_audio_input_token" in MODEL_CARDS[model_name]:
            usd_per_input_token = MODEL_CARDS[model_name]["usd_per_audio_input_token"]
        else:
            usd_per_input_token = MODEL_CARDS[model_name].get("usd_per_input_token")
        
        if usd_per_input_token is None:
            raise ValueError(
                f"Model '{model_name}' has usd_per_input_token=None in MODEL_CARDS. "
                f"This model may not support cost estimation. Model card: {MODEL_CARDS[model_name]}"
            )

        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["overall"] / 100.0)

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
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            image_operation=self.is_image_op(),
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])
    
class SemanticGroupByOp(AggregateOp):
    """
    Implementation of a semantic GroupBy operator using LLMs. This operator groups records by a set 
    of fields and applies aggregation functions to each group using an LLM to determine the groups.
    """
    def __init__(self, gby_fields: list[str] | list[dict], agg_fields: list[str] | list[dict], agg_funcs: list[str], 
                 model: Model | None = None, prompt_strategy: PromptStrategy = PromptStrategy.AGG, 
                 reasoning_effort: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Store original field specifications (may be dicts or strings)
        self.gby_fields_spec = gby_fields
        self.agg_fields_spec = agg_fields
        
        # Extract field names for backward compatibility
        self.gby_fields = [f['name'] if isinstance(f, dict) else f for f in gby_fields]
        self.agg_fields = [f['name'] if isinstance(f, dict) else f for f in agg_fields]
        
        self.agg_funcs = agg_funcs
        self.model = model
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        
        # Initialize the generator for LLM calls
        self.generator = Generator(self.model, self.prompt_strategy, self.reasoning_effort, self.api_base)

    def __str__(self):
        op = super().__str__()
        op += f"    Group-by Fields: {self.gby_fields}\n"
        op += f"    Agg. Fields: {self.agg_fields}\n"
        op += f"    Agg. Funcs: {self.agg_funcs}\n"
        op += f"    Model: {self.model.value}\n"
        op += f"    Prompt Strategy: {self.prompt_strategy}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "gby_fields": self.gby_fields, 
            "agg_fields": self.agg_fields, 
            "agg_funcs": self.agg_funcs,
            "model": self.model.value,
            "prompt_strategy": self.prompt_strategy.value,
            "reasoning_effort": self.reasoning_effort,
            **id_params
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "gby_fields": self.gby_fields, 
            "agg_fields": self.agg_fields, 
            "agg_funcs": self.agg_funcs,
            "model": self.model,
            "prompt_strategy": self.prompt_strategy,
            "reasoning_effort": self.reasoning_effort,
            **op_params
        }
    
    def get_model_name(self) -> str:
        return self.model.value

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        """
        Compute naive cost estimates for the semantic group by operation using an LLM.
        """
        # estimate number of input and output tokens
        est_num_input_tokens = NAIVE_EST_NUM_INPUT_TOKENS * source_op_cost_estimates.cardinality
        est_num_output_tokens = NAIVE_EST_NUM_OUTPUT_TOKENS * NAIVE_EST_NUM_GROUPS

        # get est. of conversion time per record from model card
        model_name = self.model.value
        model_conversion_time_per_record = MODEL_CARDS[model_name]["seconds_per_output_token"] * est_num_output_tokens

        # get est. of conversion cost (in USD) per record from model card
        # Check for audio models first
        if "usd_per_audio_input_token" in MODEL_CARDS[model_name]:
            usd_per_input_token = MODEL_CARDS[model_name]["usd_per_audio_input_token"]
        else:
            usd_per_input_token = MODEL_CARDS[model_name].get("usd_per_input_token")
        
        if usd_per_input_token is None:
            raise ValueError(
                f"Model '{model_name}' has usd_per_input_token=None in MODEL_CARDS. "
                f"This model may not support cost estimation. Model card: {MODEL_CARDS[model_name]}"
            )
        
        model_conversion_usd_per_record = (
            usd_per_input_token * est_num_input_tokens
            + MODEL_CARDS[model_name]["usd_per_output_token"] * est_num_output_tokens
        )

        # estimate quality of output based on the strength of the model being used
        quality = (MODEL_CARDS[model_name]["overall"] / 100.0)

        return OperatorCostEstimates(
            cardinality=NAIVE_EST_NUM_GROUPS,
            time_per_record=model_conversion_time_per_record,
            cost_per_record=model_conversion_usd_per_record,
            quality=quality,
        )

    def __call__(self, candidates: list[DataRecord]) -> DataRecordSet:
        """
        Execute the semantic group by operation on the given candidates using a two-phase approach:
        Phase 1: LLM assigns each record to a group (MAP)
        Phase 2: Apply aggregation functions to each group (REDUCE)
        
        Args:
            candidates: List of DataRecords to group and aggregate
            
        Returns:
            DataRecordSet containing one DataRecord per group with aggregated values
        """
        start_time = time.time()
        
        # Handle empty input
        if len(candidates) == 0:
            return DataRecordSet([], [])
        
        # Use LLM to assign each record to a semantic group
        group_assignments, gen_stats = self._assign_groups_llm(candidates)
        
        # Group candidates by their assigned group labels and compute aggregations
        # Using the same approach as ApplyGroupByOp but with LLM-determined groups
        agg_state = {}
        for candidate, group_label in zip(candidates, group_assignments):
            # Use group_label as the group key (tuple with single element)
            group = (group_label,)
            
            # Initialize aggregation state for new groups
            if group not in agg_state:
                state = []
                for fun in self.agg_funcs:
                    state.append(ApplyGroupByOp.agg_init(fun))
            else:
                state = agg_state[group]
            
            # Merge values from this candidate into the aggregation state
            for i in range(0, len(self.agg_funcs)):
                fun = self.agg_funcs[i]
                if not hasattr(candidate, self.agg_fields[i]):
                    raise TypeError(f"SemanticGroupByOp record missing expected field {self.agg_fields[i]}")
                field = getattr(candidate, self.agg_fields[i])
                state[i] = ApplyGroupByOp.agg_merge(fun, state[i], field)
            
            agg_state[group] = state
        
        # Create output DataRecords (one per group)
        drs = []
        record_op_stats_lst = []
        
        # Get the output field names from the output schema
        output_field_names = [f for f in self.output_schema.model_fields.keys() if f not in self.gby_fields]
        
        for group_key in agg_state:
            # Build aggregated data item for this group
            data_item = {}
            
            # Add group-by field value (extract from tuple)
            data_item[self.gby_fields[0]] = group_key[0]
            
            # Add aggregation results (using agg_final to compute final values)
            vals = agg_state[group_key]
            for i in range(0, len(vals)):
                agg_func = self.agg_funcs[i]
                output_field_name = output_field_names[i]
                v = ApplyGroupByOp.agg_final(agg_func, vals[i])
                data_item[output_field_name] = v
            
            # Create the DataRecord for this group
            data_item_obj = self.output_schema(**data_item)
            dr = DataRecord.from_agg_parents(data_item_obj, parent_records=candidates)
            drs.append(dr)
            
            # Create RecordOpStats for this group
            # Cost is from LLM group assignment only (aggregation is free)
            record_op_stats = RecordOpStats(
                record_id=dr._id,
                record_parent_ids=dr._parent_ids,
                record_source_indices=dr._source_indices,
                record_state=dr.to_dict(include_bytes=False),
                full_op_id=self.get_full_op_id(),
                logical_op_id=self.logical_op_id or "semantic-groupby",
                op_name=self.op_name(),
                time_per_record=(time.time() - start_time) / len(agg_state),
                cost_per_record=gen_stats.cost_per_record / len(agg_state),
                model_name=self.get_model_name(),
                input_fields=self.get_input_fields(),
                generated_fields=list(self.output_schema.model_fields.keys()),
                total_input_tokens=gen_stats.total_input_tokens,
                total_output_tokens=gen_stats.total_output_tokens,
                total_input_cost=gen_stats.total_input_cost,
                total_output_cost=gen_stats.total_output_cost,
                llm_call_duration_secs=gen_stats.llm_call_duration_secs,
                fn_call_duration_secs=gen_stats.fn_call_duration_secs,
                total_llm_calls=gen_stats.total_llm_calls,
                total_embedding_llm_calls=gen_stats.total_embedding_llm_calls,
                op_details={k: str(v) for k, v in self.get_id_params().items()},
            )
            record_op_stats_lst.append(record_op_stats)
        
        return DataRecordSet(drs, record_op_stats_lst)
    
    def _assign_groups_llm(self, candidates: list[DataRecord]) -> tuple[list[str], any]:
        """
        Phase 1: Use LLM to assign each candidate to a semantic group.
        
        Args:
            candidates: List of DataRecords to classify into groups
            
        Returns:
            Tuple of (list of group labels, generation stats)
        """
        # Create a schema that just extracts the group-by field
        # Use the description from the field spec if available
        from palimpzest.core.lib.schemas import create_schema_from_fields
        
        first_gby_spec = self.gby_fields_spec[0]
        if isinstance(first_gby_spec, dict):
            field_desc = first_gby_spec["desc"]
            field_name = first_gby_spec['name']
            field_type = first_gby_spec.get('type', str)
        else:
            field_desc = f"The semantic category for {first_gby_spec}"
            field_name = first_gby_spec
            field_type = str
        
        groupby_schema = create_schema_from_fields([
            {"name": field_name, "type": field_type, "desc": field_desc}
        ])
        
        # Process candidates to extract group labels
        group_labels = []
        total_stats = GenerationStats()
        
        # Get input fields - but only use the groupby field to avoid image detection issues
        # Since ImageFilepath is just an alias for str, passing all string fields causes
        # the prompt factory to try to open them as image files
        input_fields = [self.gby_fields[0]]  # Only pass the groupby field

        fields = {self.gby_fields[0]: str}
        
        # Build the aggregation instruction that includes the field descriptions from field spec 
        # This tells the LLM HOW to categorize/group the values semantically
        agg_instruction = f"Categorize this record into a semantic group based on the field '{field_name}' Return the category name (one of those specified in '{field_desc}'s)"
        
        print(f"\nSemanticGroupByOp: Processing {len(candidates)} records for group assignment...")
        print(f"  Grouping instruction: {agg_instruction}")
        for idx, candidate in enumerate(candidates):
            # Show progress every 10 records
            if idx % 10 == 0:
                print(f"  Processing record {idx+1}/{len(candidates)}...")
            
            # Ask LLM to categorize the record according to the field description
            gen_kwargs = {
                "project_cols": input_fields,
                "output_schema": groupby_schema,
                "agg_instruction": agg_instruction
            }
            
            field_answers, _, gen_stats, _ = self.generator(candidate, fields, **gen_kwargs)
            
            # Extract the group label - field_answers returns dict with field->list mapping
            field_answer = field_answers.get(self.gby_fields[0])
            if field_answer is None or not isinstance(field_answer, list) or len(field_answer) == 0:
                group_label = "unknown"
            else:
                group_label = field_answer[0]
            group_labels.append(group_label)
            
            # Accumulate stats
            total_stats += gen_stats
        
        print(f"  Completed! Found {len(set(group_labels))} unique groups from {len(candidates)} records")
        return group_labels, total_stats

    def hierarchical_groupby(
        self,
        candidates: list[DataRecord],
        groupby_fields: list[list[str | dict]],
        agg_fields: list[list[str | dict]],
        agg_funcs: list[list[str]],
        model: Model = None,
        prompt_strategy: PromptStrategy = PromptStrategy.AGG,
        reasoning_effort: str | None = None,
    ) -> dict:
        """
        Perform hierarchical (nested) semantic groupby operations using LLMs.

        At each intermediate level the LLM assigns group labels to the original records
        (without aggregation) so that inner levels operate on the same raw records.
        The final level runs a full semantic groupby with aggregation.

        Args:
            candidates: Input DataRecords.
            groupby_fields: List of lists of field specs per level.
            agg_fields: List of lists of aggregate field specs per level.
            agg_funcs: List of lists of aggregation function names per level.
            model: Optional LLM model override (falls back to self.model).
            prompt_strategy: Prompt strategy (defaults to AGG).
            reasoning_effort: Optional reasoning effort override.

        Returns:
            A DataRecordSet for a single level, or a nested dict for multiple levels.
        """
        from palimpzest.core.lib.schemas import create_groupby_schema_from_fields

        assert len(groupby_fields) == len(agg_fields) == len(agg_funcs), \
            "groupby_fields, agg_fields, and agg_funcs must all have the same length"

        def normalize_fields(fields):
            out = []
            for f in fields:
                if isinstance(f, str):
                    out.append({'name': f, 'desc': f'Group by {f}', 'type': str})
                else:
                    out.append(f)
            return out

        _model = model or self.model
        _prompt_strategy = prompt_strategy or self.prompt_strategy
        _reasoning_effort = reasoning_effort or self.reasoning_effort

        def run_level(candidates, level):
            gby_specs = normalize_fields(groupby_fields[level])
            agg_specs = normalize_fields(agg_fields[level])
            funcs = agg_funcs[level]
            gby_names = [s['name'] for s in gby_specs]
            agg_names = [s['name'] for s in agg_specs]
            output_schema = create_groupby_schema_from_fields(gby_names, agg_names)
            op = SemanticGroupByOp(
                gby_fields=gby_specs,
                agg_fields=agg_specs,
                agg_funcs=funcs,
                model=_model,
                prompt_strategy=_prompt_strategy,
                reasoning_effort=_reasoning_effort,
                output_schema=output_schema,
                input_schema=self.input_schema,
            )
            if level == len(groupby_fields) - 1:
                return op(candidates)
            # Intermediate: LLM assigns labels, original records are forwarded
            group_labels, _ = op._assign_groups_llm(candidates)
            outer_groups = {}
            for candidate, label in zip(candidates, group_labels):
                key = (label,) if not isinstance(label, tuple) else label
                outer_groups.setdefault(key, []).append(candidate)
            return {key: run_level(grp, level + 1) for key, grp in outer_groups.items()}

        return run_level(candidates, 0)