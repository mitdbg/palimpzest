"""GV have to separate cost estimator from sthe StatsProcessor because of circular dependency.
Operators needs statprocessors, defined in stats.py
CostEstimators needs PhysicalPlans which needs operators.py

Notes: 2. why is _estimate_plan cost a hidden function? Probably we should only expose the single parameter function
       that takes a single plan, and have the Execution call it in a for loop on a list of plans.
"""

from __future__ import annotations

from palimpzest.constants import GPT_4_MODEL_CARD, Model, MODEL_CARDS, QueryStrategy
from palimpzest.planner import PhysicalPlan
from palimpzest.utils import getModels

import palimpzest as pz

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import math

# TYPE DEFINITIONS
SampleExecutionData = Dict[str, Any] # TODO: dataclass?

class CostEstimator:
    """
    This class takes in a list of SampleExecutionData and exposes a function which uses this data
    to perform cost estimation on a list of physical plans.
    """
    def __init__(self, sample_execution_data: List[SampleExecutionData] = []):
        # construct full dataset of samples
        self.sample_execution_data_df = (
            pd.DataFrame(sample_execution_data)
            if len(sample_execution_data) > 0
            else None
        )

        # TODO: come up w/solution for listing operators by name due to circular import
        # determine the set of operators which may use a distinct model
        self.MODEL_OPERATORS = ["LLMFilter", "LLMConvert"]

    # GV: Does it make sense to have static and private method?
    # MR: My understanding, which may be wrong / overly-simplified, is that:
    #     "static" == "doesn't rely on self or cls", and
    #     "private" == "does not need to be called from outside of this class"
    #
    #     so I thought this fit? but maybe I'm wrong
    @staticmethod
    def _est_time_per_record(
        op_df: pd.DataFrame, model_name: Optional[str] = None, agg: str = "mean"
    ) -> float:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the `op_time` column.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return model_df["op_time"].agg(agg=agg).iloc[0]

        # compute aggregate
        return op_df["op_time"].agg(agg=agg).iloc[0]

    @staticmethod
    def _est_cost_per_record(
        op_df: pd.DataFrame, model_name: Optional[str] = None, agg: str = "mean"
    ) -> float:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the sum of the `input_usd` and `output_usd` columns.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return (
                    (model_df["input_usd"] + model_df["output_usd"])
                    .agg(agg=agg)
                    .iloc[0]
                )

        # # adjust cost from sample model to this model
        # this_model_usd_per_input_token = MODEL_CARDS[model_name]['usd_per_input_token']
        # model_to_usd_per_input_token = {
        #     model_name: model_card['usd_per_input_token']
        #     for model_name, model_card in MODEL_CARDS.items()
        # }
        # this_model_usd_per_output_token = MODEL_CARDS[model_name]['usd_per_output_token']
        # model_to_usd_per_output_token = {
        #     model_name: model_card['usd_per_output_token']
        #     for model_name, model_card in MODEL_CARDS.items()
        # }
        # df.loc[:, 'adj_input_usd'] = df.apply(lambda row: row['input_usd'] * (this_model_usd_per_input_token / model_to_usd_per_input_token[row['model_name']]), axis=1)
        # df.loc[:, 'adj_output_usd'] = df.apply(lambda row: row['output_usd'] * (this_model_usd_per_output_token / model_to_usd_per_output_token[row['model_name']]), axis=1)

        # # compute average combined input/output usd spent
        # return (df['adj_input_usd'] + df['adj_output_usd']).agg(agg=agg).iloc[0]
        return 0
        #TODO This code is not working? There are None in the total_input_cost columns
        return (op_df["total_input_cost"] + op_df["total_output_cost"]).agg(agg=agg).iloc[0]

    @staticmethod
    def _est_tokens_per_record(
        op_df: pd.DataFrame, model_name: Optional[str] = None, agg: str = "mean"
    ) -> Tuple[float, float]:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the `total_input_tokens` and `total_output_tokens` columns.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return (
                    model_df["total_input_tokens"].agg(agg=agg).iloc[0],
                    model_df["total_output_tokens"].agg(agg=agg).iloc[0],
                )

        # compute aggregate
        return (
            op_df["total_input_tokens"].agg(agg=agg).iloc[0],
            op_df["total_output_tokens"].agg(agg=agg).iloc[0],
        )

    @staticmethod
    def _est_cardinality(op_df: pd.DataFrame, model_name: Optional[str] = None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the number of
        rows output by the operation.

        NOTE: right now, this should only be used by the ApplyGroupByOp as a way to gauge the
        number of output groups. Using this to estimate, e.g. the cardinality of a filter,
        convert, or base scan will lead to wildly inaccurate results because the absolute value
        of these cardinalities will simply be a reflection of the sample size.

        For those operations, we use the `_est_selectivity` function to estimate the operator's
        selectivity, which we can apply to an est. of the operator's input cardinality.
        """
        return op_df.shape[0] / len(op_df.plan_id.unique())

    @staticmethod
    def _est_selectivity(
        df: pd.DataFrame, op_df: pd.DataFrame, model_name: Optional[str] = None
    ) -> float:
        """
        Given sample cost data observations for the plan and a specific operation, compute
        the ratio of records between this operator and its source operator.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_op_df = op_df[op_df.model_name == model_name]
            if not model_op_df.empty:
                num_input_records = model_op_df.shape[0]

                # get subset of records that were the source to this operator
                op_name = str(model_op_df.op_name.iloc[0])
                num_output_records = None
                if "filter" in op_name:
                    num_output_records = model_op_df.passed_filter.sum()
                else:
                    op_ids = model_op_df.op_id.unique().tolist()
                    num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

                return num_output_records / num_input_records

        # otherwise average selectivity across all ops
        num_input_records = op_df.shape[0]

        # get subset of records that were the source to this operator
        op_name = str(op_df.op_name.iloc[0])
        num_output_records = None
        if "filter" in op_name:
            num_output_records = op_df.passed_filter.sum()
        else:
            op_ids = op_df.op_id.unique().tolist()
            num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

        return num_output_records / num_input_records

    @staticmethod
    def _est_quality(op_df: pd.DataFrame, model_name: Optional[str] = None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the an estimate
        of the quality of its outputs by using GPT-4 as a champion model.
        """
        # get unique set of records
        record_uuids = op_df.record_uuid.unique()

        # compute GPT-4's answer (per-record) across all models; fall-back to most common answer if GPT-4 is not present
        record_uuid_to_answer = {}
        for record_uuid in record_uuids:
            record_df = op_df[op_df.record_uuid == record_uuid]
            gpt4_most_common_answer = record_df[
                record_df.model_name == Model.GPT_4.value
            ].answer.mode()

            if not gpt4_most_common_answer.empty:
                record_uuid_to_answer[record_uuid] = gpt4_most_common_answer.iloc[0]
            else:
                record_uuid_to_answer[record_uuid] = record_df.answer.mode().iloc[0]

        def _is_correct(row):
            # simple equality check suffices for filter
            if "filter" in row["op_name"]:
                return row["answer"] == row["accepted_answer"]

            # otherwise, check equality on a per-key basis
            try:
                # we'll measure recal on accepted_answer, as extraneous info is often not an issue
                answer = row["answer"]
                accepted_answer = row["accepted_answer"]

                tp = 0
                for key, value in accepted_answer.items():
                    if key in answer and answer[key] == value:
                        tp += 1

                return tp / len(accepted_answer)

            except Exception as e:
                print(f"WARNING: error decoding answer or accepted_answer: {str(e)}")
                return False

        # compute accepted answers and clean all answers
        pd.options.mode.chained_assignment = None  # turn off copy warnings
        op_df.loc[:, "accepted_answer"] = op_df.record_uuid.apply(
            lambda uuid: record_uuid_to_answer[uuid]
        )
        op_df.loc[:, "correct"] = op_df.apply(lambda row: _is_correct(row), axis=1)

        # get subset of observations for model_name and estimate quality w/fraction of answers that match accepted answer
        model_df = (
            op_df[op_df.model_name == model_name]
            if model_name is not None
            else op_df[op_df.model_name.isna()]
        )

        est_quality = (
            model_df.correct.sum() / model_df.shape[0]
            if not model_df.empty
            else (
                op_df.correct.sum() / op_df.shape[0]
                if not op_df.empty
                else MODEL_CARDS[model_name]["MMLU"] / 100.0
            )
        )

        return est_quality

    def _compute_operator_estimates(self) -> Optional[Dict[str, Any]]:
        """
        Compute per-operator estimates of runtime, cost, and quality.
        """
        # if we don't have sample execution data, we cannot compute per-operator estimates
        if self.sample_execution_data_df is None:
            return None

        # get the set of operator ids for which we have sample data
        op_ids = self.sample_execution_data_df.op_id.unique()

        # compute estimates of runtime, cost, and quality (and intermediates like cardinality) for every operator
        operator_estimates = {}
        for op_id in op_ids:
            # filter for subset of sample execution data related to this operation
            op_df = self.sample_execution_data_df[
                self.sample_execution_data_df.op_id == op_id
            ]

            # skip computing an estimate if we didn't capture any sampling data for this operator
            # (this can happen if/when upstream filter operation(s) filter out all records) 
            if op_df.empty:
                continue

            # initialize estimates
            estimates = {}

            # get the op_name for this operation
            op_name = str(op_df.op_name.iloc[0])
            if op_name in self.MODEL_OPERATORS:
                # compute estimates per-model, and add None which forces computation of avg. across all models
                models = getModels(include_vision=True) + [None]
                estimates = {model: None for model in models}
                for model in models:
                    model_name = model.value if model is not None else None
                    est_tokens = CostEstimator._est_tokens_per_record(op_df, model_name=model_name)
                    model_estimates = {
                        "time_per_record": CostEstimator._est_time_per_record(op_df, model_name=model_name),
                        "cost_per_record": CostEstimator._est_cost_per_record(op_df, model_name=model_name),
                        "input_tokens": est_tokens[0],
                        "output_tokens": est_tokens[1],
                        "selectivity": CostEstimator._est_selectivity(self.sample_execution_data_df, op_df, model_name=model_name),
                        "quality": CostEstimator._est_quality(op_df, model_name=model_name),
                    }
                    estimates[model_name] = model_estimates
            
            elif op_name in ["MarshalAndScanDataOp", "CacheScanDataOp", "LimitScanOp", "ApplyCountAggregateOp", "ApplyAverageAggregateOp"]:
                estimates = {
                    "time_per_record": CostEstimator._est_time_per_record(op_df),
                }

            elif op_name in ["ApplyGroupByOp"]:
                estimates = {
                    "time_per_record": CostEstimator._est_time_per_record(op_df),
                    "cardinality": CostEstimator._est_cardinality(op_df),
                }

            operator_estimates[op_id] = estimates
        
        return operator_estimates

    def _estimate_plan_cost(self, physical_plan: PhysicalPlan, sample_op_estimates: Optional[Dict[str, Any]]) -> None:
        # initialize dictionary w/estimates for entire plan
        plan_estimates = {"total_time": 0.0, "total_cost": 0.0, "quality": 0.0}

        op_estimates, source_op_estimates = None, None
        for op in physical_plan.operators:
            # get identifier for operation which is unique within sentinel plan but consistent across sentinels
            op_id = op.get_op_id()

            # initialize estimates of operator metrics based on naive (but sometimes precise) logic
            op_estimates = (
                op.naiveCostEstimates()
                if isinstance(op, pz.MarshalAndScanDataOp) or isinstance(op, pz.CacheScanDataOp)
                else op.naiveCostEstimates(source_op_estimates)
            )

            # if we have sample execution data, update naive estimates with more informed ones
            if sample_op_estimates is not None and op_id in sample_op_estimates:
                if isinstance(op, pz.MarshalAndScanDataOp) or isinstance(op, pz.CacheScanDataOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.ApplyGroupByOp):
                    op_estimates.cardinality = sample_op_estimates[op_id]["cardinality"]
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.ApplyCountAggregateOp) or isinstance(op, pz.ApplyAverageAggregateOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.LimitScanOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
            
                elif isinstance(op, pz.NonLLMFilter):
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id]["cost_per_record"]

                elif isinstance(op, pz.HardcodedConvert):
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.LLMFilter):
                    model_name = op.model.value
                    # TODO: account for scenario where model_name does not have samples but another model does
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                    op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]
                
                elif isinstance(op, pz.LLMConvert):
                    model_name = op.model.value
                    # TODO: account for scenario where model_name does not have samples but another model does
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                    op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]

                    # TODO: if code synth. fails, this will turn into ConventionalQuery calls to GPT-3.5,
                    #       which would wildly mess up estimate of time and cost per-record
                    # do code synthesis adjustment
                    if op.query_strategy in [
                        QueryStrategy.CODE_GEN_WITH_FALLBACK,
                        QueryStrategy.CODE_GEN,
                    ]:
                        op_estimates.time_per_record = 1e-5
                        op_estimates.cost_per_record = 1e-4
                        op_estimates.quality = op_estimates.quality * (GPT_4_MODEL_CARD["code"] / 100.0)

                    # token reduction adjustment
                    if op.token_budget is not None and op.token_budget < 1.0:
                        input_tokens = op.token_budget * sample_op_estimates[op_id][model_name]["input_tokens"]
                        output_tokens = sample_op_estimates[op_id][model_name]["output_tokens"]
                        op_estimates.cost_per_record = (
                            MODEL_CARDS[op.model.value]["usd_per_input_token"] * input_tokens
                            + MODEL_CARDS[op.model.value]["usd_per_output_token"] * output_tokens
                        )
                        op_estimates.quality = op_estimates.quality * math.sqrt(math.sqrt(op.token_budget))

                else:
                    raise Exception("Unknown operator")

            # NOTE: a slightly more accurate thing to do would be to estimate the time_per_record based on the
            #       *input* cardinality to the operator and multiply by the estimated input cardinality.
            # update plan estimates
            plan_estimates["total_time"] += op_estimates.time_per_record * op_estimates.cardinality
            plan_estimates["total_cost"] += op_estimates.cost_per_record * op_estimates.cardinality
            plan_estimates["quality"] *= op_estimates.quality

            # update source_op_estimates
            source_op_estimates = op_estimates

        # set the plan's estimates
        physical_plan.total_time = plan_estimates["total_time"]
        physical_plan.total_cost = plan_estimates["total_cost"]
        physical_plan.quality = plan_estimates["quality"]

    def estimate_plan_costs(self, physical_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
        """
        Estimate the cost of each physical plan by making use of the sample execution data
        provided to the CostEstimator. The plan cost, runtime, and quality are set as attributes
        on each physical plan and the updated set of physical plans is returned.
        """
        operator_estimates = self._compute_operator_estimates()

        for physical_plan in physical_plans:
            self._estimate_plan_cost(physical_plan, operator_estimates)

        return physical_plans
