from palimpzest.constants import PlanType
from palimpzest.cost_estimator import CostEstimator
from palimpzest.dataclasses import ExecutionStats
from palimpzest.planner import LogicalPlanner, PhysicalPlanner
from palimpzest.policy import Policy
from palimpzest.execution import (
    ExecutionEngine,
    PipelinedParallelExecutionEngine,
    PipelinedSingleThreadExecutionEngine,
    SequentialSingleThreadExecutionEngine,
)
from palimpzest.sets import Set

import os
import shutil
import time


class NoSentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """

    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # Always delete cache
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)
        cache = self.datadir.getCacheService()
        cache.rmCache()

        self.set_source_dataset_id(dataset)

        # construct the CostEstimator with any sample execution data we've gathered
        cost_estimator = CostEstimator(source_dataset_id=self.source_dataset_id)

        # construct the pruning strategy if specified
        pruning_strategy = self.get_pruning_strategy(cost_estimator)

        # (re-)initialize logical and physical planner
        logical_planner = LogicalPlanner(self.nocache, verbose=self.verbose)
        physical_planner = PhysicalPlanner(
            pruning_strategy=pruning_strategy,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_conventional_query=self.allow_conventional_query,
            allow_model_selection=self.allow_model_selection,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            verbose=self.verbose,
        )

        # enumerate all possible physical plans
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)

        # estimate the cost of each plan
        for physical_plan in all_physical_plans:
            total_time, total_cost, quality = cost_estimator.estimate_plan_cost(physical_plan)
            physical_plan.total_time = total_time
            physical_plan.total_cost = total_cost
            physical_plan.quality = quality

        # deduplicate plans with identical cost estimates
        plans = physical_planner.deduplicate_plans(all_physical_plans)

        # select pareto frontier of plans
        final_plans = physical_planner.select_pareto_optimal_plans(plans)

        # for experimental evaluation, we may want to include baseline plans
        if self.include_baselines:
            final_plans = physical_planner.add_baseline_plans(final_plans)

        if self.min_plans is not None and len(final_plans) < self.min_plans:
            final_plans = physical_planner.add_plans_closest_to_frontier(final_plans, plans, self.min_plans)

        # choose best plan and execute it
        final_plan = policy.choose(plans)
        all_records, plan_stats = self.execute_plan(plan=final_plan,
                                               plan_type=PlanType.FINAL,
                                               plan_idx=0,
                                               max_workers=self.max_workers)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats={plan_stats.plan_id: plan_stats},
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=plan_stats.total_plan_cost,
        )

        return all_records, final_plan, execution_stats


class SequentialSingleThreadNoSentinelExecution(NoSentinelExecutionEngine, SequentialSingleThreadExecutionEngine):
    """
    This class performs non-sample based execution while executing plans in a sequential, single-threaded fashion.
    """
    pass


class PipelinedSingleThreadNoSentinelExecution(NoSentinelExecutionEngine, PipelinedSingleThreadExecutionEngine):
    """
    This class performs non-sample based execution while executing plans in a pipelined, single-threaded fashion.
    """
    pass


class PipelinedParallelNoSentinelExecution(NoSentinelExecutionEngine, PipelinedParallelExecutionEngine):
    """
    This class performs non-sample based execution while executing plans in a pipelined, parallel fashion.
    """
    pass
