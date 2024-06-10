from palimpzest.datamanager import DataDirectory
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.profiler import CostOptimizer
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import os
import shutil


class Execute:
    @staticmethod
    def run_sentinel_plan(plan: PhysicalPlan, plan_idx=0, verbose=False): # TODO: does ThreadPoolExecutor support tuple unpacking for arguments?
        # display the plan output
        if verbose:
            print("----------------------")
            print(f"Sentinel Plan {plan_idx}:")
            plan.printPlan()
            print("---")

        # run the plan
        records, stats = plan.execute()

        return records, stats

    @classmethod
    def run_sentinel_plans(
        cls, sentinel_plans: List[PhysicalPlan], verbose: bool = False
    ):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        all_sample_execution_data, return_records = [], []
        with ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
            results = list(
                executor.map(
                    Execute.run_sentinel_plan,
                    [(plan, idx, verbose) for idx, plan in enumerate(sentinel_plans)],
                )
            )

            # write out result dict and samples collected for each sentinel
            sentinel_records, sentinel_stats = zip(*results)
            for records, stats, plan in zip(sentinel_records, sentinel_stats, sentinel_plans):
                # aggregate sentinel est. data
                sample_execution_data = plan.getExecutionData()
                all_sample_execution_data.extend(sample_execution_data)

                # set return_records to be records from champion model
                champion_model_name = getChampionModelName()

                # find champion model plan records and add those to all_records
                if champion_model_name in plan.getPlanModelNames():
                    return_records = records

        return all_sample_execution_data, return_records # TODO: make sure you capture cost of sentinel plans.


    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int=20,
        nocache: bool=False,
        include_baselines: bool=False,
        min_plans: Optional[int] = None,
        verbose: bool = False,
    ):
        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if nocache:
            dspy_cache_dir = os.path.join(
                os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/"
            )
            if os.path.exists(dspy_cache_dir):
                shutil.rmtree(dspy_cache_dir)

            # remove codegen samples from previous dataset from cache
            cache = DataDirectory().getCacheService()
            cache.rmCache()

        # only run sentinels if there isn't a cached result already
        uid = (
            dataset.universalIdentifier()
        )  # TODO: I think we may need to get uid from source?
        run_sentinels = nocache or not DataDirectory().hasCachedAnswer(uid)

        sentinel_plans, sample_execution_data, sentinel_records = [], [], []
        if run_sentinels:
            # initialize logical and physical planner
            logical_planner = LogicalPlanner(nocache)
            physical_planner = PhysicalPlanner(num_samples, scan_start_idx=0)

            # get sentinel plans
            for logical_plan in logical_planner.generate_plans(dataset, sentinels=True):
                for sentinel_plan in physical_planner.generate_plans(logical_plan, sentinels=True):
                    sentinel_plans.append(sentinel_plan)

            # run sentinel plans
            sample_execution_data, sentinel_records = cls.run_sentinel_plans(
                sentinel_plans, verbose
            )

        # (re-)initialize logical and physical planner
        scan_start_idx = num_samples if run_sentinels else 0
        logical_planner = LogicalPlanner(nocache)
        physical_planner = PhysicalPlanner(scan_start_idx=scan_start_idx)

        # NOTE: in the future we may use operator_estimates below to limit the number of plans
        #       that we need to consider during plan generation. I.e., we may be able to save time
        #       by pre-computing the set of viable models / execution strategies at each operator
        #       based on the sample execution data we get.
        # 
        # enumerate all possible physical plans
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)

        # TODO: still WIP
        # construct the CostOptimizer with any sample execution data we've gathered
        cost_optimizer = CostOptimizer(sample_execution_data)

        # estimate the cost of each plan
        plans = cost_optimizer.estimate_plan_costs(all_physical_plans)

        # deduplicate plans with identical cost estimates
        plans = physical_planner.deduplicate_plans(plans)

        # select pareto frontier of plans
        final_plans = physical_planner.select_pareto_optimal_plans(plans)

        # for experimental evaluation, we may want to include baseline plans
        if include_baselines:
            final_plans = physical_planner.add_baseline_plans(final_plans)

        if min_plans is not None and len(final_plans) < min_plans:
            final_plans = physical_planner.add_plans_closest_to_frontier(final_plans, plans, min_plans)

        # choose best plan and execute it
        plan = policy.choose(plans) # TODO: have policies accept PhysicalPlan

        # display the plan output
        if verbose:
            print("----------------------")
            print(f"Final Plan:")
            plan.printPlan()
            print("---")

        # run the plan
        new_records, stats = plan.execute() # TODO: Still WIP

        all_records = sentinel_records + new_records

        return all_records, plan, stats
