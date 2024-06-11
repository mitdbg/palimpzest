import time
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.datamanager import DataDirectory
from palimpzest.operators.filter import FilterOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from .cost_estimator import CostEstimator
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

# backwards compatability for users who are still on Python 3.9
try:
    from itertools import pairwise
except:
    from more_itertools import pairwise
from palimpzest.dataclasses import OperatorStats, PlanStats, SampleExecutionData

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import os
import shutil


class Execute:
    @staticmethod
    def run_sentinel_plan(cls, plan: PhysicalPlan, plan_idx=0, verbose=False): # TODO: does ThreadPoolExecutor support tuple unpacking for arguments?
        # display the plan output
        if verbose:
            print("----------------------")
            print(f"Sentinel Plan {plan_idx}:")
            plan.printPlan()
            print("---")

        # run the plan
        records, stats = cls.execute(plan)

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
                sample_execution_data = cls.getSampleExecutionData(plan, stats)
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

        # construct the CostEstimator with any sample execution data we've gathered
        cost_optimizer = CostEstimator(sample_execution_data)

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
        plan = policy.choose(plans)

        # display the plan output
        if verbose:
            print("----------------------")
            print(f"Final Plan:")
            plan.printPlan()
            print("---")

        # run the plan
        new_records, stats = cls.execute(plan) # TODO: Still WIP

        all_records = sentinel_records + new_records

        return all_records, plan, stats


    
    # GV I am open to making it a private function if need be
    @staticmethod
    def execute(cls, plan: PhysicalPlan, stats: PlanStats=None):
        """Execute the plan."""
        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id())
        for op_idx, op in enumerate(plan.operators):
            op_id = op.physical_op_id()
            plan_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here

        # initialize list of output records
        output_records = []

        # TODO: execution issues I still need to resolve:
        #  - need to force upstream execution to complete for agg., groupby, and parallel operators
        #    -  eventually we could allow for pipelining and not block on parallel operators, but pre-SIGMOD let's keep that behavior intact
        # iterate over records from the datasource operator
        datasource_operator = plan.operators[0]
        for record, record_op_stats in datasource_operator:
            plan_stats[datasource_operator.physical_op_id()] += record_op_stats

            # apply sequence of subsequent operators
            filtered = False
            for operator in plan.operators[1:]:
                record, record_op_stats = operator(record)

                # add record_op_stats to appropriate operator
                plan_stats[datasource_operator.physical_op_id()] += record_op_stats

                # TODO: confirm if this isinstance will work with sub-class operations
                if isinstance(operator, FilterOp) and not record._passed_filter:
                    filtered = True
                    break

            if not filtered:
                output_records.append(record)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats

    @staticmethod
    def getSampleExecutionData(cls, plan_stats:PhysicalPlan) -> List[Dict[str, Any]]:
        """Compute and return all sample execution data collected by this plan so far."""
        # construct table of observation data from sample batch of processed records
        sample_execution_data, source_op_id = [], None
        for op_id, operator_stats in plan_stats.operator_stats.items():
            # append observation data for each record
            for record_op_stats in operator_stats.record_op_stats_lst:
                # compute minimal observation which is supported by all operators
                # TODO: one issue with this setup is that cache_scans of previously computed queries
                #       may not match w/these observations due to the diff. op_name
                
                # GV Readability changes
                observation_arguments = {
                    "passed_filter": (
                        record_op_stats.record_state["_passed_filter"]
                        if "_passed_filter" in record_op_stats.record_state
                        else None
                    ),
                    "model_name": (
                        record_op_stats.op_details["model_name"]
                        if "model_name" in record_op_stats.op_details
                        else None
                    ),
                    "filter_str": (
                        record_op_stats.op_details["filter_str"]
                        if "filter_str" in record_op_stats.op_details
                        else None
                    ),
                    "input_fields_str": (
                        "-".join(sorted(record_op_stats.op_details["input_fields"]))
                        if "input_fields" in record_op_stats.op_details
                        else None
                    ),
                    "generated_fields_str": (
                        "-".join(sorted(record_op_stats.op_details["generated_fields"]))
                        if "generated_fields" in record_op_stats.op_details
                        else None
                    ),
                    "total_input_tokens": (
                        record_op_stats.record_stats["total_input_tokens"]
                        if "total_input_tokens" in record_op_stats.record_stats
                        else None
                    ),
                    "total_output_tokens": (
                        record_op_stats.record_stats["total_output_tokens"]
                        if "total_output_tokens" in record_op_stats.record_stats
                        else None
                    ),
                    "total_input_cost": (
                        record_op_stats.record_stats["total_input_cost"]
                        if "total_input_cost" in record_op_stats.record_stats
                        else None
                    ),
                    "total_output_cost": (
                        record_op_stats.record_stats["total_output_cost"]
                        if "total_output_cost" in record_op_stats.record_stats
                        else None
                    ),
                }

                # return T/F for filter
                if "_passed_filter" in record_op_stats.record_state:
                    observation_arguments["answer"] = record_op_stats.record_state["_passed_filter"]
                else:
                    answer = {}
                    # return key->value mapping for generated fields for induce
                    if "generated_fields" in record_op_stats.op_details:
                        for field in record_op_stats.op_details["generated_fields"]:
                            answer[field] = record_op_stats.record_state[field]
                    observation_arguments["answer"] = answer

                observation = SampleExecutionData(**observation_arguments)

                # add observation to list of observations
                sample_execution_data.append(observation)

            # update source_op_id
            source_op_id = op_id

        return sample_execution_data