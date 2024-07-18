from palimpzest.constants import PlanType
from palimpzest.cost_estimator import CostEstimator
from palimpzest.dataclasses import ExecutionStats, RecordOpStats
from palimpzest.execution import ExecutionEngine
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from concurrent.futures import ThreadPoolExecutor
from typing import List

import os
import shutil
import time


class ConfidenceIntervalPruningExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """
    def get_seed_plans(self, dataset: Set) -> List[PhysicalPlan]:
        """
        Get an initial set of seed plans to feed into the exploration phase.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")

    def run_exploration_phase(self, plans: List[PhysicalPlan]):
        """
        Generate sample execution data.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")

    def prune_plans(self, sample_execution_data: List[RecordOpStats]):
        """
        Estimate the total time, cost, and quality of each plan using the sample execution data
        and filter out any plans whose upper bound on their confidence interval is below some threshold
        (e.g. the user policy constraint, the lower bound of a pareto-dominant plan, etc.) 
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")

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

        # initialize variables
        final_plan, all_records, all_plan_stats = None, [], []

        # get a set of seed plans
        plans = self.get_seed_plans(dataset)

        # run exploration until we settle on a final plan
        while final_plan is None:
            # run exploration phase in which we gather more sample execution data;
            # this phase may also return new output records and plan stats
            new_sample_execution_data, new_records, new_plan_stats = self.run_exploration_phase(plans)
            all_records.extend(new_records)
            all_plan_stats.extend(new_plan_stats)

            # determine whether or not we've converged to a single final plan
            final_plan, plans = self.prune_plans(new_sample_execution_data)

            # if we run out of records to process; break

        # once we have a final plan; execute it
        final_records, final_plan_stats = self.execute_plan(plan=final_plan,
                                               plan_type=PlanType.FINAL, 
                                               plan_idx=0,
                                               max_workers=self.max_workers)

        # add sentinel records and plan stats (if captured) to plan execution data
        all_records.extend(final_records)
        all_plan_stats.extend(final_plan_stats)
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats={plan_stats.plan_id: plan_stats for plan_stats in all_plan_stats},
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, all_plan_stats))),
        )

        return all_records, final_plan, execution_stats


class ConfidenceIntervalPruningSentinelExecution(ConfidenceIntervalPruningExecutionEngine):

    def get_seed_plans(self, dataset: Set) -> List[PhysicalPlan]:
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

        return sentinel_plans

    def run_exploration_phase(self, plans: List[PhysicalPlan]):
        # compute number of plans
        num_sentinel_plans = len(plans)

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
                      for idx, plan in enumerate(plans)],
                )
            )

        # split results into per-plan records and plan stats
        sentinel_records, sentinel_plan_stats = zip(*results)

        # get champion model
        champion_model_name = getChampionModelName()

        # process results to get sample execution data and sentinel plan stats
        all_sample_execution_data, return_records = [], []
        for records, plan_stats, plan in zip(sentinel_records, sentinel_plan_stats, plans):
            # aggregate sentinel est. data
            for operator_stats in plan_stats.operator_stats.values():
                all_sample_execution_data.extend(operator_stats.record_op_stats_lst)

            # find champion model plan records and add those to all_records
            if champion_model_name in plan.getPlanModelNames():
                return_records = records

        return all_sample_execution_data, return_records, sentinel_plan_stats

    def prune_plans(self, dataset: Set, sample_execution_data: List[RecordOpStats]):
        """
        Estimate the total time, cost, and quality of each plan using the sample execution data
        and filter out any plans whose upper bound on their confidence interval is below some threshold
        (e.g. the user policy constraint, the lower bound of a pareto-dominant plan, etc.) 
        """
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
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            compile_start = time.time()
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)
            compile_end = time.time()
            print(f"PHYSICAL COMPILATION TIME: {compile_end - compile_start}")

        import pdb; pdb.set_trace()

        # estimate the cost of each plan
        for physical_plan in all_physical_plans:
            total_time, total_cost, quality = cost_estimator.estimate_plan_cost(physical_plan)
            physical_plan.total_time = total_time
            physical_plan.total_cost = total_cost
            physical_plan.quality = quality

        import pdb; pdb.set_trace()