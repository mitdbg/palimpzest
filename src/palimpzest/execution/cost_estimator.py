"""GV have to separate cost estimator from sthe StatsProcessor because of circular dependency.
Operators needs statprocessors, defined in stats.py
CostEstimators needs PhysicalPlans which needs operators.py

Notes: 1. why is it called costOptimizer if all it does is Estimate costs? Should rename to CostEstimator?
       2. why is _estimate_plan cost a hidden function? Probably we should only expose the single parameter function
       that takes a single plan, and have the Execution call it in a for loop on a list of plans.
"""

import palimpzest as pz
from palimpzest.planner import PhysicalPlan
from typing import Any, Dict, List, Optional, Tuple, Union
from palimpzest.profiler.stats import StatsProcessor

import pandas as pd

# TYPE DEFINITIONS
SampleExecutionData = Dict[str, Any] # TODO: dataclass?

class CostOptimizer:
    """
    This class takes in a list of SampleExecutionData and exposes a function which uses this data
    to perform cost estimation on a list of physical plans.
    """
    def __init__(self, sample_execution_data: List[SampleExecutionData] = []):
        # construct full dataset of samples
        self.sample_execution_data_df = (
            pd.DataFrame(sample_execution_data)
            if len(sample_execution_data) > 0
            else pd.DataFrame()
        )

    def _compute_operator_estimates(self) -> Optional[Dict[str, Any]]:
        """
        Compute per-operator estimates of runtime, cost, and quality.
        """
        # TODO:
        # - switch to iterating over op_ids present in sample execution data
        # - move StatsProcessor estimation fcn.'s into this class
        # - produce identical operator_estimates dict. to the one we previously had
        # for now, each sentinel uses the same logical plan, so we can simply use the first one
        sentinel_plan = sentinel_plans[0]

        # construct full dataset of samples
        sample_exec_data_df = (
            pd.DataFrame(self.sample_execution_data)
            if self.sample_execution_data is not None and self.sample_execution_data != []
            else pd.DataFrame()
        )

        # compute estimates of runtime, cost, and quality (and intermediates like cardinality) for every operator
        operator_estimates = {}
        for op in sentinel_plan.operators:
            # get unique identifier for operator
            op_id = op.op_id()
            # - base / cache scan: dataset identifier
            # - convert: output schema (generated fields)
            # - filter: filter condition
            # - limit: <doesn't matter; just pass through and set output cardinality>
            # - count / avg: <doesn't matter; just pass through and set output cardinality>
            # - gby: <doesn't matter; just pass through and set output cardinality>

            # filter for subset of sample execution data related to this operation
            op_df = (
                sample_exec_data_df.query(op_id)
                if not sample_exec_data_df.empty
                else pd.DataFrame()
            )

            estimates = {}
            if op_df.empty:
                estimates = op.naiveCostEstimates() # TODO
            else:
                # get model name for this operation (if applicable)
                model_name = repr(getattr(op, "model", None))

                estimates = StatsProcessor.compute_operator_estimates(sample_exec_data_df)

                est_input_tokens, est_output_tokens = StatsProcessor.est_tokens_per_record(op_df, model_name)
                estimates = {
                    "time_per_record": StatsProcessor.est_time_per_record(op_df, model_name),
                    "cost_per_record": StatsProcessor.est_cost_per_record(op_df, model_name),
                    "input_tokens_per_record": est_input_tokens,
                    "output_tokens_per_record": est_output_tokens,
                    "selectivity": StatsProcessor.est_selectivity(op_df, model_name),
                    "quality": StatsProcessor.est_quality(sample_exec_data_df, op_df, model_name),
                }

            operator_estimates[op_id] = estimates

    def _estimate_plan_cost(physical_plan: PhysicalPlan, sample_op_estimates: Optional[Dict[str, Any]]) -> None:
        # initialize dictionary w/estimates for entire plan
        plan_estimates = {"total_time": 0.0, "total_cost": 0.0, "quality": 0.0}

        op_estimates, source_op_estimates = None, None
        for op in physical_plan:
            # get identifier for operation which is unique within sentinel plan but consistent across sentinels
            op_id = op.physical_op_id()

            # initialize estimates of operator metrics based on naive (but sometimes precise) logic
            op_estimates = (
                op.naiveCostEstimates()
                if isinstance(op, pz.operators.MarshalAndScanDataOp) or isinstance(op, pz.operators.CacheScanDataOp)
                else op.naiveCostEstimates(source_op_estimates)
            )

            # if we have sample execution data, update naive estimates with more informed ones
            if sample_op_estimates is not None and op_id in sample_op_estimates:
                if isinstance(op, pz.operators.MarshalAndScanDataOp) or isinstance(op, pz.operators.CacheScanDataOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.operators.ApplyGroupByOp):
                    op_estimates.cardinality = sample_op_estimates[op_id]["cardinality"]
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.operators.ApplyCountAggregateOp) or isinstance(op, pz.operators.ApplyAverageAggregateOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]

                elif isinstance(op, pz.operators.LimitScanOp):
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
            
                elif isinstance(op, pz.operators.NonLLMFilter):
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id]["time_per_record"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id]["cost_per_record"]

                elif isinstance(op, pz.operators.LLMFilter):
                    model_name = op.model.value
                    # TODO: account for scenario where model_name does not have samples but another model does
                    op_estimates.cardinality = source_op_estimates.cardinality * sample_op_estimates[op_id][model_name]["selectivity"]
                    op_estimates.time_per_record = sample_op_estimates[op_id][model_name]["time_per_record"]
                    op_estimates.cost_per_record = sample_op_estimates[op_id][model_name]["cost_per_record"]
                    op_estimates.quality = sample_op_estimates[op_id][model_name]["quality"]

                # TODO: convert operators

            # NOTE: a slightly more accurate thing to do would be to estimate the time_per_record based on the
            #       *input* cardinality to the operator and multiply by the estimated input cardinality.
            # update plan estimates
            plan_estimates["total_time"] += op_estimates.time_per_record * op_estimates.cardinality
            plan_estimates["total_cost"] += op_estimates.cost_per_record * op_estimates.cardinality
            plan_estimates["quality"] *= op_estimates.quality

            # update source_op_estimates
            source_op_estimates = op_estimates

        # set the plan's estimates
        physical_plan.estimates = plan_estimates

    def estimate_plan_costs(self, physical_plans: List[PhysicalPlan]) -> List[PhysicalPlan]:
        """
        Estimate the cost of each physical plan by making use of the sample execution data
        provided to the CostOptimizer. The plan cost, runtime, and quality are set as attributes
        on each physical plan and the updated set of physical plans is returned.
        """
        operator_estimates = self._compute_operator_estimates()

        for physical_plan in physical_plans:
            self._estimate_plan_cost(physical_plan, operator_estimates)

        return physical_plans


