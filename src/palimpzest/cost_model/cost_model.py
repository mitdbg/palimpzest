from __future__ import annotations

from palimpzest.constants import Cardinality, GPT_4_MODEL_CARD, MODEL_CARDS
from palimpzest.dataclasses import ExpressionCost, OperatorCostEstimates, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.operators import PhysicalOperator
from palimpzest.utils import getChampionModelName, getModels

import palimpzest as pz

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import scipy.stats as stats
import math

# NOTE: the answer.mode() call(s) inside of _est_quality() throw a UserWarning when there are multiple
#       answers to a convert with the same mode. This is because pandas tries to sort the answers
#       before returning them, but since answer is a column of dicts the '<' operator fails on dicts.
#       For now, we can simply ignore the warning b/c we pick an answer at random anyways if their are
#       multiple w/the same count, but in the future we may want to cast the 'dict' --> 'str' or compute
#       the mode on a per-field basis.
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class CostModel:
    """
    This class takes in a list of RecordOpStats and exposes a function which uses this data
    to perform cost estimation on a list of physical plans.
    """
    def __init__(self, source_dataset_id: str, sample_execution_data: List[RecordOpStats] = [], confidence_level: float = 0.95):
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

        # reference to data directory
        self.datadir = DataDirectory()

        # compute per-operator estimates
        self.operator_estimates = self._compute_operator_estimates()

        # set confidence level for CI estimates
        self.conf_level = confidence_level

    def _compute_ci(self, sample_mean: float, n_samples: int, std_dev: float) -> Tuple[float, float]:
        """
        Compute confidence interval (for non-proportion quantities) given the sample mean, number of samples,
        and sample std. deviation at the CostModel's given confidence level. We use a t-distribution for
        computing the interval as many sample estimates in PZ may have few samples.
        """
        ci = stats.t.interval(
            confidence=self.conf_level,  # Confidence level
            df=n_samples - 1,            # Degrees of freedom
            loc=sample_mean,             # Sample mean
            scale=std_dev,               # Standard deviation estimate
        )
        return ci

    def _compute_proportion_ci(self, sample_prop: float, n_samples: int) -> Tuple[float, float]:
        """
        Compute confidence interval for proportion quantities (i.e. selectivity) given the sample proportion
        and the number of samples. We use the normal distribution for computing the interval here, for reasons
        summarized by this post: https://stats.stackexchange.com/a/411727.
        """
        scaling_factor = math.sqrt((sample_prop * (1 - sample_prop)) / n_samples)
        ci = stats.norm.interval(
            confidence=self.conf_level,  # Confidence level
            loc=sample_prop,             # Sample proportion
            scale=scaling_factor,        # Scaling factor
        )
        return ci

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
                    model_df["cost_per_record"].agg(agg=agg).iloc[0]
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

        return op_df["cost_per_record"].agg(agg=agg).iloc[0]

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
                if "filter" in op_name.lower():
                    num_output_records = model_op_df.passed_filter.sum()
                else:
                    op_ids = model_op_df.op_id.unique().tolist()
                    plan_ids = model_op_df.plan_id.unique().tolist()
                    num_output_records = df[df.source_op_id.isin(op_ids) & df.plan_id.isin(plan_ids)].shape[0]

                return num_output_records / num_input_records

        # otherwise average selectivity across all ops
        num_input_records = op_df.shape[0]

        # get subset of records that were the source to this operator
        op_name = str(op_df.op_name.iloc[0])
        num_output_records = None
        if "filter" in op_name.lower():
            num_output_records = op_df.passed_filter.sum()
        else:
            op_ids = op_df.op_id.unique().tolist()
            num_output_records = df[df.source_op_id.isin(op_ids)].shape[0]

        return num_output_records / num_input_records

    def _compute_quality(self, row):
        # compute accuracy for filter
        if "filter" in row["op_name"].lower():
            return int(row["answer"] == row["accepted_answer"])

        # otherwise, compute recall on a per-key basis
        try:
            # we'll measure recall on accepted_answer, as extraneous info is often not an issue
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
        record_ids = op_df.record_id.unique()

        # get champion model name
        champion_model_name = getChampionModelName()

        # compute champion's answer (per-record) across all models; fall-back to most common answer if champion is not present
        record_id_to_answer = {}
        for record_id in record_ids:
            record_df = op_df[op_df.record_id == record_id]
            champion_most_common_answer = record_df[
                record_df.model_name == champion_model_name
            ].answer.mode()
            all_models_most_common_answer = record_df.answer.mode()

            if not champion_most_common_answer.empty:
                record_id_to_answer[record_id] = champion_most_common_answer.iloc[0]
            elif not all_models_most_common_answer.empty:
                record_id_to_answer[record_id] = all_models_most_common_answer.iloc[0]
            else:
                record_id_to_answer[record_id] = None

        # compute accepted answers and clean all answers
        pd.options.mode.chained_assignment = None  # turn off copy warnings
        op_df.loc[:, "accepted_answer"] = op_df.record_id.apply(
            lambda id: record_id_to_answer[id]
        )
        op_df.loc[:, "quality"] = op_df.apply(lambda row: self._compute_quality(row), axis=1)

        # get subset of observations for model_name and estimate quality w/fraction of answers that match accepted answer
        model_df = (
            op_df[op_df.model_name == model_name]
            if model_name is not None
            else op_df[op_df.model_name.isna()]
        )

        est_quality = (
            model_df.quality.sum() / model_df.shape[0]
            if not model_df.empty
            else (
                op_df.quality.sum() / op_df.shape[0]
                if not op_df.empty
                else MODEL_CARDS[model_name]["MMLU"] / 100.0
            )
        )

        return est_quality

    def _compute_operator_estimates(self) -> Optional[Dict[str, Any]]:
        """
        Compute per-operator estimates of runtime, cost, and quality.

        TODO: include lower and upper bounds
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
            model_name = op_df.model_name.iloc[0] if op_df.model_name.iloc[0] is not None else None
            op_name = str(op_df.op_name.iloc[0])
            if model_name is not None:
                # compute estimates per-model, and add None which forces computation of avg. across all models
                model_names = [m.value for m in getModels(include_vision=True)] + [None]
                # model_names = op_df.model_name.unique().tolist()
                estimates = {model_name: None for model_name in model_names}
                for model_name in model_names:
                    est_tokens = self._est_tokens_per_record(op_df, model_name=model_name)
                    model_estimates = {
                        "time_per_record": self._est_time_per_record(op_df, model_name=model_name),
                        "cost_per_record": self._est_cost_per_record(op_df, model_name=model_name),
                        "total_input_tokens": est_tokens[0],
                        "total_output_tokens": est_tokens[1],
                        "selectivity": self._est_selectivity(self.sample_execution_data_df, op_df, model_name=model_name),
                        "quality": self._est_quality(op_df, model_name=model_name),
                    }
                    estimates[model_name] = model_estimates

            # TODO pre-compute lists of op_names in groups
            elif op_name in ["NonLLMFilter"]:
                est_tokens = self._est_tokens_per_record(op_df)
                estimates = {
                    "time_per_record": self._est_time_per_record(op_df),
                    "cost_per_record": self._est_cost_per_record(op_df),
                    "selectivity": self._est_selectivity(self.sample_execution_data_df, op_df),
                    "quality": self._est_quality(op_df, model_name=model_name),
                }

            elif op_name in ["MarshalAndScanDataOp", "CacheScanDataOp", "LimitScanOp", "CountAggregateOp", "AverageAggregateOp"]:
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

    def __call__(self, operator: PhysicalOperator, source_op_estimates: Optional[OperatorCostEstimates]=None) -> ExpressionCost:
        # get identifier for operation which is unique within sentinel plan but consistent across sentinels
        op_id = operator.get_op_id()

        # initialize estimates of operator metrics based on naive (but sometimes precise) logic
        if isinstance(operator, pz.MarshalAndScanDataOp):
            # get handle to DataSource and pre-compute its size (number of records)
            datasource = self.datadir.getRegisteredDataset(self.source_dataset_id)
            dataset_type = self.datadir.getRegisteredDatasetType(self.source_dataset_id)
            datasource_len = len(datasource)
            datasource_memsize = datasource.getSize()

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naiveCostEstimates(source_op_estimates,
                                                    input_cardinality=datasource.cardinality,
                                                    input_record_size_in_bytes=datasource_memsize/datasource_len,
                                                    dataset_type=dataset_type)

        elif isinstance(operator, pz.CacheScanDataOp):
            datasource = self.datadir.getCachedResult(operator.cachedDataIdentifier)
            datasource_len = len(datasource)
            datasource_memsize = datasource.getSize()

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naiveCostEstimates(source_op_estimates,
                                                    input_cardinality=Cardinality.ONE_TO_ONE,
                                                    input_record_size_in_bytes=datasource_memsize/datasource_len)

        else:
            op_estimates = operator.naiveCostEstimates(source_op_estimates)

        # if we have sample execution data, update naive estimates with more informed ones
        sample_op_estimates = self.operator_estimates
        if sample_op_estimates is not None and op_id in sample_op_estimates:
            if isinstance(operator, pz.MarshalAndScanDataOp) or isinstance(operator, pz.CacheScanDataOp):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.ApplyGroupByOp):
                op_estimates.cardinality = sample_op_estimates[op_id]["cardinality"]
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.time_per_record_lower_bound = sample_op_estimates[op_id]["time_per_record_lower_bound"]
                op_estimates.time_per_record_upper_bound = sample_op_estimates[op_id]["time_per_record_upper_bound"]

            elif isinstance(operator, pz.CountAggregateOp) or isinstance(operator, pz.AverageAggregateOp):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

            elif isinstance(operator, pz.LimitScanOp):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

            elif isinstance(operator, pz.NonLLMFilter):
                op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id]["selectivity"]
                op_estimates.cost_per_record = sample_op_estimates[op_id]["cost_per_record"]

            elif isinstance(operator, pz.LLMFilter):
                model_name = operator.model.value
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]

            elif isinstance(operator, pz.LLMConvert):
                # TODO: EVEN BETTER: do similarity match (e.g. largest param intersection, more exotic techniques);
                #       another heuristic: logical_op_id-->subclass_physical_op_id-->specific_physical_op_id-->most_param_match_physical_op_id
                # TODO: instead of [op_id][model_name] --> [logical_op_id][physical_op_id]
                # NOTE: code synthesis does not have a model attribute
                model_name = operator.model.value if hasattr(operator, "model") else None
                op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]

                # NOTE: if code synth. fails, this will turn into ConventionalQuery calls to GPT-3.5,
                #       which would wildly mess up estimate of time and cost per-record
                # do code synthesis adjustment
                if isinstance(operator, pz.CodeSynthesisConvert):
                    op_estimates.time_per_record = 1e-5
                    op_estimates.cost_per_record = 1e-4
                    op_estimates.quality = op_estimates.quality * (GPT_4_MODEL_CARD["code"] / 100.0)

                # token reduction adjustment
                if isinstance(operator, pz.TokenReducedConvert):
                    total_input_tokens = operator.token_budget * sample_op_estimates[op_id][model_name]["total_input_tokens"]
                    total_output_tokens = sample_op_estimates[op_id][model_name]["total_output_tokens"]
                    op_estimates.cost_per_record = (
                        MODEL_CARDS[operator.model.value]["usd_per_input_token"] * total_input_tokens
                        + MODEL_CARDS[operator.model.value]["usd_per_output_token"] * total_output_tokens
                    )
                    op_estimates.quality = op_estimates.quality * math.sqrt(math.sqrt(operator.token_budget))

            else:
                raise Exception("Unknown operator")

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * source_op_estimates.cardinality
        op_cost = op_estimates.cost_per_record * source_op_estimates.cardinality
        op_quality = op_estimates.quality * source_op_estimates.quality

        # compute bounds on estimates for this operator
        op_time_lower_bound = sample_op_estimates[op_id]

        # create and return ExpressionCost object
        expr_cost_obj = ExpressionCost(
            cost=op_cost,
            time=op_time,
            quality=op_quality,
            op_estimates=op_estimates,
            cost_lower_bound=,
            cost_upper_bound=,
            time_lower_bound=,
            time_upper_bound=,
            quality_lower_bound=,
            quality_upper_bound=,
        )

        return expr_cost_obj
