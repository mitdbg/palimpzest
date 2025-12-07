from __future__ import annotations

import logging
import warnings

import pandas as pd

from palimpzest.constants import NAIVE_BYTES_PER_RECORD
from palimpzest.core.models import OperatorCostEstimates, PlanCost, SentinelPlanStats
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, MarshalAndScanDataOp, ScanPhysicalOp

warnings.simplefilter(action='ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class BaseCostModel:
    """
    This base class contains the interface/abstraction that every CostModel must implement
    in order to work with the Optimizer. In brief, the Optimizer expects the CostModel to
    make a prediction about the runtime, cost, and quality of a physical operator.
    """
    def __init__(self):
        """
        CostModel constructor; the arguments for individual CostModels may vary depending
        on the assumptions they make about the prevalance of historical execution data
        and online vs. batch execution settings.
        """
        pass

    def get_costed_full_op_ids(self) -> set[str]:
        """
        Return the set of full op ids which the cost model has cost estimates for.
        """
        raise NotImplementedError("Calling get_costed_full_op_ids from abstract method")

    def __call__(self, operator: PhysicalOperator) -> PlanCost:
        """
        The interface exposed by the CostModel to the Optimizer. Subclasses may require
        additional arguments in order to make their predictions.
        """
        raise NotImplementedError("Calling __call__ from abstract method")


class SampleBasedCostModel:
    """
    """
    def __init__(
        self,
        sentinel_plan_stats: SentinelPlanStats | None = None,
        verbose: bool = False,
        exp_name: str | None = None,
    ):
        # store verbose argument
        self.verbose = verbose

        # store experiment name if one is provided
        self.exp_name = exp_name

        # construct cost, time, quality, and selectivity matrices for each operator set;
        self.operator_to_stats = self._compute_operator_stats(sentinel_plan_stats)
        self.costed_full_op_ids = None if self.operator_to_stats is None else set([
            full_op_id
            for _, full_op_id_to_stats in self.operator_to_stats.items()
            for full_op_id in full_op_id_to_stats
        ])

        # if there is a logical operator with no samples; add all of its op ids to costed_full_op_ids;
        # this will lead to the cost model applying the naive cost estimates for all physical op ids
        # in this logical operator (I think?)
        # TODO

        logger.info(f"Initialized SampleBasedCostModel with verbose={self.verbose}")
        logger.debug(f"Initialized SampleBasedCostModel with params: {self.__dict__}")

    def get_costed_full_op_ids(self):
        return self.costed_full_op_ids

    def _compute_operator_stats(self, sentinel_plan_stats: SentinelPlanStats | None) -> dict:
        logger.debug("Computing operator statistics")
        # if no stats are provided, simply return None
        if sentinel_plan_stats is None:
            return None

        # flatten the nested dictionary of execution data and pull out fields relevant to cost estimation
        execution_record_op_stats = []
        for unique_logical_op_id, full_op_id_to_op_stats in sentinel_plan_stats.operator_stats.items():
            logger.debug(f"Computing operator statistics for logical_op_id: {unique_logical_op_id}")
            # flatten the execution data into a list of RecordOpStats
            op_set_execution_data = [
                record_op_stats
                for _, op_stats in full_op_id_to_op_stats.items()
                for record_op_stats in op_stats.record_op_stats_lst
            ]

            # add entries from execution data into matrices
            for record_op_stats in op_set_execution_data:
                record_op_stats_dict = {
                    "unique_logical_op_id": unique_logical_op_id,
                    "full_op_id": record_op_stats.full_op_id,
                    "record_id": record_op_stats.record_id,
                    "record_parent_ids": record_op_stats.record_parent_ids,
                    "cost_per_record": record_op_stats.cost_per_record,
                    "time_per_record": record_op_stats.time_per_record,
                    "quality": record_op_stats.quality,
                    "passed_operator": record_op_stats.passed_operator,
                    "source_indices": record_op_stats.record_source_indices,
                    "op_details": record_op_stats.op_details,
                    "answer": record_op_stats.answer,
                    "op_name": record_op_stats.op_name,
                }
                execution_record_op_stats.append(record_op_stats_dict)

        # convert flattened execution data into dataframe
        operator_stats_df = pd.DataFrame(execution_record_op_stats)

        # for each full_op_id, compute its average cost_per_record, time_per_record, selectivity, and quality
        operator_to_stats = {}
        for unique_logical_op_id, logical_op_df in operator_stats_df.groupby("unique_logical_op_id"):
            logger.debug(f"Computing operator statistics for unique_logical_op_id: {unique_logical_op_id}")
            operator_to_stats[unique_logical_op_id] = {}

            for full_op_id, physical_op_df in logical_op_df.groupby("full_op_id"):
                # compute the number of input records processed by this operator; use source_indices for scan operator(s)
                num_source_records = (
                    physical_op_df.record_parent_ids.apply(tuple).nunique()
                    if not physical_op_df.record_parent_ids.isna().all()
                    else physical_op_df.source_indices.apply(tuple).nunique()
                )

                # compute selectivity; for filters this may be 1.0 on smalle samples;
                # always put something slightly less than 1.0 to ensure that filters are pushed down when possible
                selectivity = physical_op_df.passed_operator.sum() / num_source_records
                op_name = physical_op_df.op_name.iloc[0].lower()
                if selectivity == 1.0 and "filter" in op_name:
                    selectivity -= 1e-3

                # compute quality; if all qualities are None then this will be NaN
                quality = physical_op_df.quality.mean()

                # set operator stats for this physical operator
                operator_to_stats[unique_logical_op_id][full_op_id] = {
                    "cost": physical_op_df.cost_per_record.mean(),
                    "time": physical_op_df.time_per_record.mean(),
                    "quality": 1.0 if pd.isna(quality) else quality,
                    "selectivity": selectivity,
                }

        logger.debug(f"Done computing operator statistics for {len(operator_to_stats)} operators!")
        return operator_to_stats

    def _compute_naive_plan_cost(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None, right_source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
        # get identifier for operator which is unique within sentinel plan but consistent across sentinels
        full_op_id = operator.get_full_op_id()
        logger.debug(f"Calling __call__ for {str(operator)} with full_op_id: {full_op_id}")

        # initialize estimates of operator metrics based on naive (but sometimes precise) logic
        if isinstance(operator, MarshalAndScanDataOp):
            # get handle to scan operator and pre-compute its size (number of records)
            datasource_len = len(operator.datasource)

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naive_cost_estimates(source_op_estimates, input_record_size_in_bytes=NAIVE_BYTES_PER_RECORD)

        elif isinstance(operator, ContextScanOp):
            source_op_estimates = OperatorCostEstimates(
                cardinality=1.0,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

            op_estimates = operator.naive_cost_estimates(source_op_estimates)

        elif isinstance(operator, JoinOp):
            op_estimates = operator.naive_cost_estimates(source_op_estimates, right_source_op_estimates)

        else:
            op_estimates = operator.naive_cost_estimates(source_op_estimates)

        # compute estimates for this operator
        est_input_cardinality = (
            source_op_estimates.cardinality * right_source_op_estimates.cardinality
            if isinstance(operator, JoinOp)
            else source_op_estimates.cardinality
        )
        op_time = op_estimates.time_per_record * est_input_cardinality
        op_cost = op_estimates.cost_per_record * est_input_cardinality
        op_quality = op_estimates.quality

        # create and return PlanCost object for this op's statistics
        op_plan_cost = PlanCost(
            cost=op_cost,
            time=op_time,
            quality=op_quality,
            op_estimates=op_estimates,
        )
        logger.debug(f"Done calling __call__ for {str(operator)} with full_op_id: {full_op_id}")
        logger.debug(f"Plan cost: {op_plan_cost}")

        return op_plan_cost

    def __call__(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None, right_source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
        # for non-sentinel execution, we use naive estimates
        full_op_id = operator.get_full_op_id()
        unique_logical_op_id = operator.unique_logical_op_id
        if self.operator_to_stats is None or unique_logical_op_id not in self.operator_to_stats:
            return self._compute_naive_plan_cost(operator, source_op_estimates, right_source_op_estimates)

        # NOTE: some physical operators may not have any sample execution data in this cost model;
        #       these physical operators are filtered out of the Optimizer, thus we can assume that
        #       we will have execution data for each operator passed into __call__; nevertheless, we
        #       still perform a sanity check
        # look up physical and logical op ids associated with this physical operator
        physical_op_to_stats = self.operator_to_stats.get(unique_logical_op_id)
        assert physical_op_to_stats is not None, f"No execution data for logical operator: {str(operator)}"
        assert physical_op_to_stats.get(full_op_id) is not None, f"No execution data for physical operator: {str(operator)}"
        logger.debug(f"Calling __call__ for {str(operator)}")

        # look up stats for this operation
        est_cost_per_record = self.operator_to_stats[unique_logical_op_id][full_op_id]["cost"]
        est_time_per_record = self.operator_to_stats[unique_logical_op_id][full_op_id]["time"]
        est_quality = self.operator_to_stats[unique_logical_op_id][full_op_id]["quality"]
        est_selectivity = self.operator_to_stats[unique_logical_op_id][full_op_id]["selectivity"]

        # create source_op_estimates for scan operators if they are not provided
        if isinstance(operator, ScanPhysicalOp):
            # get handle to scan operator and pre-compute its size (number of records)
            datasource_len = len(operator.datasource)

            source_op_estimates = OperatorCostEstimates(
                cardinality=datasource_len,
                time_per_record=0.0,
                cost_per_record=0.0,
                quality=1.0,
            )

        # generate new set of OperatorCostEstimates
        est_input_cardinality = (
            source_op_estimates.cardinality * right_source_op_estimates.cardinality
            if isinstance(operator, JoinOp)
            else source_op_estimates.cardinality
        )
        op_estimates = OperatorCostEstimates(
            cardinality=est_selectivity * est_input_cardinality,
            time_per_record=est_time_per_record,
            cost_per_record=est_cost_per_record,
            quality=est_quality,
        )

        # compute estimates for this operator
        op_time = op_estimates.time_per_record * est_input_cardinality
        op_cost = op_estimates.cost_per_record * est_input_cardinality
        op_quality = op_estimates.quality

        # construct and return op estimates
        plan_cost = PlanCost(cost=op_cost, time=op_time, quality=op_quality, op_estimates=op_estimates)
        logger.debug(f"Done calling __call__ for {str(operator)}")
        logger.debug(f"Plan cost: {plan_cost}")
        return plan_cost
