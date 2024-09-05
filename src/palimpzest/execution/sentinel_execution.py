from palimpzest.constants import OptimizationStrategy
from palimpzest.cost_model import CostModel
from palimpzest.dataclasses import ExecutionStats, RecordOpStats
from palimpzest.execution import (
    ExecutionEngine,
    PipelinedParallelSentinelPlanExecutor,
    SequentialSingleThreadSentinelPlanExecutor,
)
from palimpzest.optimizer import Optimizer, SentinelPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set

from typing import List

import time


class SentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """
    def __init__(self, rank: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank


    def execute_sentinel_plan(self, plan: SentinelPlan):
        """
        """
        # TODO
        pass


    def generate_sample_observations(self, dataset: Set):
        """
        This function is responsible for generating sample observation data which can be
        consumed by the CostModel. For each physical optimization and each operator, our
        goal is to capture `rank + 1` samples per optimization, where `rank` is the presumed
        low-rank of the observation matrix.

        To accomplish this, we construct a special sentinel plan using the Optimizer which is
        capable of executing any valid physical implementation of a Filter or Convert operator
        on each record.
        """
        # initialize the optimizer
        optimizer = Optimizer(
            policy=None,
            cost_model=None,
            no_cache=True,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=True,
            allow_conventional_query=True,
            allow_code_synth=True,
            allow_token_reduction=True,
            optimization_strategy=OptimizationStrategy.SENTINEL,
            sentinel_low_rank=self.rank,
        )

        # use optimizer to generate sentinel plans
        sentinel_plan = optimizer.optimize(dataset)

        # run sentinel plan
        execution_data, records, plan_stats = self.execute_sentinel_plan(sentinel_plan)

        return execution_data, records, plan_stats


    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            self.clear_cached_responses_and_examples()

        # set the source dataset id
        self.set_source_dataset_id(dataset)

        # initialize execution data, records, and plan stats
        all_execution_data, all_records, all_plan_stats = [], [], []

        # NOTE: this checks if the entire computation is cached; it will re-run
        #       the sentinels even if the computation is partially cached
        # only run sentinels if there isn't a cached result already
        uid = dataset.universalIdentifier()
        run_sentinels = self.nocache or not self.datadir.hasCachedAnswer(uid)    
        if run_sentinels:
            all_execution_data, all_records, all_plan_stats = self.generate_sample_observations(dataset, policy)
            # TODO: if the above is validation data: do not return as part of results; if it is sample data: do return;

        # construct the CostModel with any sample execution data we've gathered
        cost_model = CostModel(
            source_dataset_id=self.source_dataset_id,
            sample_execution_data=all_execution_data,
        )

        # NOTE: if we change sentinel execution to run a diverse set of plans, we may only need to update_cost_model here
        # (re-)initialize the optimizer
        optimizer = Optimizer(
            policy=policy,
            cost_model=cost_model,
            no_cache=self.nocache,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_conventional_query=self.allow_conventional_query,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            optimization_strategy=self.optimization_strategy,
        )

        # execute plan(s) according to the optimization strategy
        if self.optimization_strategy == OptimizationStrategy.OPTIMAL:
            records, plan_stats = self.execute_optimal_strategy(dataset, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        elif self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            records, plan_stats = self.execute_confidence_interval_strategy(dataset, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(all_plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))),
            plan_strs={plan_stats.plan_id: plan_stats.plan_str for plan_stats in aggregate_plan_stats.items()},
        )

        return all_records, execution_stats


class SequentialSingleThreadSentinelExecution(SentinelExecutionEngine, SequentialSingleThreadSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a sequential, single-threaded fashion.
    """
    pass


class PipelinedParallelSentinelExecution(SentinelExecutionEngine, PipelinedParallelSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    pass
