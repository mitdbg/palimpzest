from palimpzest.constants import PlanType
from palimpzest.cost_estimator import CostEstimator
from palimpzest.dataclasses import ExecutionStats
from palimpzest.execution import (
    ExecutionEngine,
    PipelinedParallelExecutionEngine,
    PipelinedSingleThreadExecutionEngine,
    SequentialSingleThreadExecutionEngine,
)
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from concurrent.futures import ThreadPoolExecutor
from typing import List

import os
import shutil
import time


class SentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """

    def run_sentinel_plans(self, sentinel_plans: List[PhysicalPlan]):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        # execute sentinel plans using any parallelism provided by the
        # user or system (self.max_workers can be set by the user, otherwise
        # it will be set to the system CPU count)
        sentinel_workers = min(self.max_workers, num_sentinel_plans)
        with ThreadPoolExecutor(max_workers=sentinel_workers) as executor:
            max_workers_per_plan = max(self.max_workers / num_sentinel_plans, 1)
            results = list(executor.map(lambda x: self.execute_plan(**x),
                    [{"plan": plan,
                      "plan_type": PlanType.SENTINEL,
                      "plan_idx": idx,
                      "max_workers": max_workers_per_plan}
                      for idx, plan in enumerate(sentinel_plans)],
                )
            )
        # results = list(map(lambda x: self.execute_plan(**x),
        #         [{"plan": plan,
        #             "plan_type": PlanType.SENTINEL,
        #             "plan_idx": idx,
        #             "max_workers": 1}
        #             for idx, plan in enumerate(sentinel_plans)],
        #     )
        # )
        # split results into per-plan records and plan stats
        sentinel_records, sentinel_plan_stats = zip(*results)

        # get champion model
        champion_model_name = getChampionModelName()

        # process results to get sample execution data and sentinel plan stats
        all_sample_execution_data, return_records = [], []
        for records, plan_stats, plan in zip(sentinel_records, sentinel_plan_stats, sentinel_plans):
            # aggregate sentinel est. data
            for operator_stats in plan_stats.operator_stats.values():
                all_sample_execution_data.extend(operator_stats.record_op_stats_lst)

            # find champion model plan records and add those to all_records
            if champion_model_name in plan.getPlanModelNames():
                return_records = records

        return all_sample_execution_data, return_records, sentinel_plan_stats


    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # TODO: we should be able to remove this w/our cache management
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            dspy_cache_dir = os.path.join(
                os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
            )
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

            # remove codegen samples from previous dataset from cache
            cache = self.datadir.getCacheService()
            cache.rmCache()

        # set the the id of the source dataset
        self.set_source_dataset_id(dataset)

        # NOTE: this checks if the entire computation is cached; it will re-run
        #       the sentinels even if the computation is partially cached
        # only run sentinels if there isn't a cached result already
        uid = dataset.universalIdentifier()
        run_sentinels = self.nocache or not self.datadir.hasCachedAnswer(uid)

        # run sentinel plans if necessary
        sample_execution_data, sentinel_records, sentinel_plan_stats = [], [], []
        if run_sentinels:
            # initialize logical and physical planner
            logical_planner = LogicalPlanner(self.nocache, sentinel=True, verbose=self.verbose)
            physical_planner = PhysicalPlanner(
                available_models=self.available_models,
                allow_bonded_query=True,
                allow_conventional_query=False,
                allow_model_selection=False,
                allow_code_synth=False,
                allow_token_reduction=False,
                verbose=self.verbose,
            )

            # use planners to generate sentinel plans
            sentinel_plans = []
            for logical_plan in logical_planner.generate_plans(dataset):
                for sentinel_plan in physical_planner.generate_plans(logical_plan):
                    sentinel_plans.append(sentinel_plan)

            # run sentinel plans
            sample_execution_data, sentinel_records, sentinel_plan_stats = self.run_sentinel_plans(sentinel_plans)

        # construct the CostEstimator with any sample execution data we've gathered
        cost_estimator = CostEstimator(source_dataset_id=self.source_dataset_id,
                                       sample_execution_data=sample_execution_data)

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
        full_compile_start = time.time()
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            compile_start = time.time()
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)
            print(f"PHYSICAL COMPILATION TIME: {time.time() - compile_start}")
        print(f"PHYSICAL COMPILATION TIME: {time.time() - full_compile_start}")

        import pdb; pdb.set_trace()

        # estimate the cost of each plan
        for physical_plan in all_physical_plans:
            total_time, total_cost, quality = cost_estimator.estimate_plan_cost(physical_plan)
            physical_plan.total_time = total_time
            physical_plan.total_cost = total_cost
            physical_plan.quality = quality

        import pdb; pdb.set_trace()

        # deduplicate plans with identical cost estimates
        plans = physical_planner.deduplicate_plans(all_physical_plans)

        import pdb; pdb.set_trace()

        # select pareto frontier of plans
        final_plans = physical_planner.select_pareto_optimal_plans(plans)

        import pdb; pdb.set_trace()

        # for experimental evaluation, we may want to include baseline plans
        if self.include_baselines:
            final_plans = physical_planner.add_baseline_plans(final_plans)

        import pdb; pdb.set_trace()

        if self.min_plans is not None and len(final_plans) < self.min_plans:
            final_plans = physical_planner.add_plans_closest_to_frontier(final_plans, plans, self.min_plans)

        import pdb; pdb.set_trace()

        # choose best plan and execute it
        final_plan = policy.choose(final_plans)
        new_records, stats = self.execute_plan(plan=final_plan,
                                               plan_type=PlanType.FINAL, 
                                               plan_idx=0,
                                               max_workers=self.max_workers)

        import pdb; pdb.set_trace()

        # add sentinel records and plan stats (if captured) to plan execution data
        all_records = sentinel_records + new_records
        all_plan_stats = sentinel_plan_stats + [stats]
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats={plan_stats.plan_id: plan_stats for plan_stats in all_plan_stats},
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, all_plan_stats))),
        )

        import pdb; pdb.set_trace()

        return all_records, final_plan, execution_stats


class SequentialSingleThreadSentinelExecution(SentinelExecutionEngine, SequentialSingleThreadExecutionEngine):
    """
    This class performs sentinel execution while executing plans in a sequential, single-threaded fashion.
    """
    pass


class PipelinedSingleThreadSentinelExecution(SentinelExecutionEngine, PipelinedSingleThreadExecutionEngine):
    """
    This class performs sentinel execution while executing plans in a pipelined, single-threaded fashion.
    """
    pass


class PipelinedParallelSentinelExecution(SentinelExecutionEngine, PipelinedParallelExecutionEngine):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    pass
