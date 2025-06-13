import logging

from palimpzest.core.data.dataclasses import ExecutionStats, SentinelPlanStats
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.query.processor.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class SentinelQueryProcessor(QueryProcessor):

    def _generate_sample_observations(self, sentinel_plan: SentinelPlan) -> SentinelPlanStats:
        """
        This function is responsible for generating sample observation data which can be
        consumed by the CostModel.

        To accomplish this, we construct a special sentinel plan using the Optimizer which is
        capable of executing any valid physical implementation of a Filter or Convert operator
        on each record.
        """
        # if we're using validation data, get the set of expected output records
        expected_outputs = {}
        for source_idx in range(len(self.val_datasource)):
            expected_output = self.val_datasource[source_idx]
            expected_outputs[source_idx] = expected_output

        # execute sentinel plan; returns sentinel_plan_stats
        return self.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, expected_outputs)

    def _create_sentinel_plan(self) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # create a new optimizer and update its strategy to SENTINEL
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_strategy(OptimizationStrategyType.SENTINEL)

        # create copy of dataset, but change its data source to the validation data source
        dataset = self.dataset.copy()
        dataset._set_data_source(self.val_datasource)

        # get the sentinel plan for the given dataset
        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan

    def execute(self) -> DataRecordCollection:
        # for now, enforce that we are using validation data; we can relax this after paper submission
        if self.val_datasource is None:
            raise Exception("Make sure you are using validation data with SentinelQueryProcessor")
        logger.info(f"Executing {self.__class__.__name__}")

        # create execution stats
        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        # create sentinel plan
        sentinel_plan = self._create_sentinel_plan()

        # generate sample execution data
        sentinel_plan_stats = self._generate_sample_observations(sentinel_plan)

        # update the execution stats to account for the work done in optimization
        execution_stats.add_plan_stats(sentinel_plan_stats)
        execution_stats.finish_optimization()

        # (re-)initialize the optimizer
        optimizer = self.optimizer.deepcopy_clean()

        # construct the CostModel with any sample execution data we've gathered
        cost_model = SampleBasedCostModel(sentinel_plan_stats, self.verbose)
        optimizer.update_cost_model(cost_model)

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, optimizer)

        # update the execution stats to account for the work to execute the final plan
        execution_stats.add_plan_stats(plan_stats)
        execution_stats.finish()

        # construct and return the DataRecordCollection
        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info("Done executing SentinelQueryProcessor")

        return result
