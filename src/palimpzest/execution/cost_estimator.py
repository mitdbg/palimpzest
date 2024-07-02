"""GV have to separate cost estimator from sthe StatsProcessor because of circular dependency.
Operators needs statprocessors, defined in stats.py
CostEstimators needs PhysicalPlans which needs operators.py

I de-statified methods within the class and made estimate_plan_cost the explicit function called on a per-plan basis by the execution engine.
"""

from __future__ import annotations
import sys

from palimpzest.constants import Cardinality, GPT_4_MODEL_CARD, Model, MODEL_CARDS, QueryStrategy
from palimpzest.dataclasses import OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
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
    def __init__(self, source_dataset_id: str, sample_execution_data: List[RecordOpStats] = []):
        # store source dataset id to help with estimating cardinalities
        self.source_dataset_id = source_dataset_id

        # construct full dataset of samples
        self.sample_execution_data_df = (
            pd.DataFrame(sample_execution_data)
            if len(sample_execution_data) > 0
            else None
        )
        # df contains a column called record_state, that sometimes contain a dict
        # we want to extract the keys from the dict and create a new column for each key
        

        # TODO: come up w/solution for listing operators by name due to circular import
        # determine the set of operators which may use a distinct model

        # reference to data directory
        self.datadir = DataDirectory()
        # TODO: Are there op_ids that repeat across plans? Otherwise we can move this into estimate_plan_cost function

        self.operator_estimates = self._compute_operator_estimates()

    def _est_time_per_record(self,
        op_df: pd.DataFrame, model_name: Optional[str] = None, agg: str = "mean"
    ) -> float:
        """
        Given sample cost data observations for a specific operation, compute the aggregate over
        the `time_per_record` column.
        """
        # use model-specific estimate if possible
        if model_name is not None:
            model_df = op_df[op_df.model_name == model_name]
            if not model_df.empty:
                return model_df["time_per_record"].agg(agg=agg).iloc[0]

        # compute aggregate
        return op_df["time_per_record"].agg(agg=agg).iloc[0]

    def _est_cost_per_record(self,
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
                    (model_df["total_input_cost"] + model_df["total_output_cost"])
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

        return (op_df["total_input_cost"] + op_df["total_output_cost"]).agg(agg=agg).iloc[0]

    def _est_tokens_per_record(self,
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

    def _est_cardinality(self,
    op_df: pd.DataFrame, model_name: Optional[str] = None) -> float:
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

    def _est_selectivity(self,
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

    def _is_correct(self, row):
        # simple equality check suffices for filter
        if "filter" in row["op_name"].lower():
            return int(row["answer"] == row["accepted_answer"])

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
            return 0
            

    def _est_quality(self, op_df: pd.DataFrame, model_name: Optional[str] = None) -> float:
        """
        Given sample cost data observations for a specific operation, compute the an estimate
        of the quality of its outputs by using GPT-4 as a champion model.
        """
        # get unique set of records
        record_uuids = op_df.record_uuid.unique()

        # compute GPT-4's answer (per-record) across all models; fall-back to most common answer if GPT-4 is not present
        record_uuid_to_answer = {}
        for record_uuid in record_uuids:
            # TODO is the fillna correct?
            record_df = op_df[op_df.record_uuid == record_uuid]
            gpt4_most_common_answer = record_df[
                record_df.model_name == Model.GPT_4.value
            ].answer.mode()

            all_models_most_common_answer = record_df.answer.mode()

            if not gpt4_most_common_answer.empty:
                record_uuid_to_answer[record_uuid] = gpt4_most_common_answer.iloc[0]
            elif not all_models_most_common_answer.empty:
                record_uuid_to_answer[record_uuid] = all_models_most_common_answer.iloc[0]
            else:
                record_uuid_to_answer[record_uuid] = ''

        # compute accepted answers and clean all answers
        pd.options.mode.chained_assignment = None  # turn off copy warnings
        op_df.loc[:, "accepted_answer"] = op_df.record_uuid.apply(
            lambda uuid: record_uuid_to_answer[uuid]
        )
        op_df.loc[:, "correct"] = op_df.apply(lambda row: self._is_correct(row), axis=1)

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
            if 'LLM' in op_name:
                # compute estimates per-model, and add None which forces computation of avg. across all models
                models = getModels(include_vision=True) + [None]
                estimates = {model: None for model in models}
                for model in models:
                    model_name = model.value if model is not None else None
                    est_tokens = self._est_tokens_per_record(op_df, model_name=model_name)
                    model_estimates = {
                        "time_per_record": self._est_time_per_record(op_df, model_name=model_name),
                        "cost_per_record": self._est_cost_per_record(op_df, model_name=model_name),
                        "input_tokens": est_tokens[0],
                        "output_tokens": est_tokens[1],
                        "selectivity": self._est_selectivity(self.sample_execution_data_df, op_df, model_name=model_name),
                        "quality": self._est_quality(op_df, model_name=model_name),
                    }
                    estimates[model_name] = model_estimates
            
            elif op_name in ["MarshalAndScanDataOp", "CacheScanDataOp", "LimitScanOp", "ApplyCountAggregateOp", "ApplyAverageAggregateOp"]:
                estimates = {
                    "time_per_record": self._est_time_per_record(op_df),
                }

            elif op_name in ["ApplyGroupByOp"]:
                estimates = {
                    "time_per_record": self._est_time_per_record(op_df),
                    "cardinality": self._est_cardinality(op_df),
                }

            operator_estimates[op_id] = estimates
        
        return operator_estimates

    def estimate_plan_cost(self, physical_plan: PhysicalPlan) -> Tuple[float, float, float]:
        # initialize dictionary w/estimates for entire plan
        plan_estimates = {"total_time": 0.0, "total_cost": 0.0, "quality": 0.0}
        sample_op_estimates = self.operator_estimates

        op_estimates, source_op_estimates = None, None
        for op in physical_plan.operators:
            # get identifier for operation which is unique within sentinel plan but consistent across sentinels
            op_id = op.get_op_id()

            # initialize estimates of operator metrics based on naive (but sometimes precise) logic
            if isinstance(op, pz.MarshalAndScanDataOp):
                # get handle to DataSource and pre-compute its size (number of records)
                datasource = self.datadir.getRegisteredDataset(self.source_dataset_id)
                datasource_len = len(datasource)
                datasource_memsize = datasource.getSize()

                source_op_estimates = OperatorCostEstimates(
                    cardinality=datasource_len,
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    quality=1.0,
                )

                op_estimates = op.naiveCostEstimates(source_op_estimates,
                                                     input_cardinality=datasource.cardinality,
                                                     input_record_size_in_bytes=datasource_memsize/datasource_len)

            elif isinstance(op, pz.CacheScanDataOp):
                datasource = self.datadir.getCachedResult(op.cachedDataIdentifier)
                datasource_len = len(datasource)
                datasource_memsize = datasource.getSize()

                source_op_estimates = OperatorCostEstimates(
                    cardinality=datasource_len,
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    quality=1.0,
                )

                op_estimates = op.naiveCostEstimates(source_op_estimates,
                                                     input_cardinality=Cardinality.ONE_TO_ONE,
                                                     input_record_size_in_bytes=datasource_memsize/datasource_len)

            else:
                op_estimates =  op.naiveCostEstimates(source_op_estimates)

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
                    # TODO check this!
                    model_name = None
                    op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]

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
        total_time = plan_estimates["total_time"]
        total_cost = plan_estimates["total_cost"]
        quality = plan_estimates["quality"]

        return total_time, total_cost, quality

    # def estimate_plan_costs(self, physical_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
    #     """
    #     Estimate the cost of each physical plan by making use of the sample execution data
    #     provided to the CostEstimator. The plan cost, runtime, and quality are set as attributes
    #     on each physical plan and the updated set of physical plans is returned.
    #     """
    #     operator_estimates = self._compute_operator_estimates()

    #     for physical_plan in physical_plans:
    #         self._estimate_plan_cost(physical_plan, operator_estimates)

    #     return physical_plans
