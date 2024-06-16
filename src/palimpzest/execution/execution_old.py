import time
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.datamanager import DataDirectory
from palimpzest.operators.physical import DataSourcePhysicalOperator, LimitScanOp
from palimpzest.operators.filter import FilterOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from .cost_estimator import CostEstimator
from palimpzest.sets import Set
from palimpzest.utils import getChampionModelName

from palimpzest.dataclasses import OperatorStats, PlanStats, SampleExecutionData

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import os
import shutil


class Execute:
    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        # Re: the num_samples parameter, I think we should consider making Execute a stateful class, otherwise we have to pass this parameter back and forth to classes to make sure it's available when we need it.
        #Alternatively, make Execute a pure facade class that just calls an Execution classes for syntax purposes.
        # See my refactor proposal in execution.py
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
        cost_estimator = CostEstimator(sample_execution_data)

        # estimate the cost of each plan
        plans = cost_estimator.estimate_plan_costs(all_physical_plans)

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

    @classmethod
    def run_sentinel_plan(cls, plan: PhysicalPlan, plan_idx=0, verbose=False): 
        # display the plan output
        if verbose:
            print("----------------------")
            print(f"Sentinel Plan {plan_idx}:")
            plan.printPlan()
            print("---")

        # run the plan
        records, plan_stats = cls.execute(plan)

        return records, plan_stats

    @classmethod
    def run_sentinel_plans(
        cls, sentinel_plans: List[PhysicalPlan], verbose: bool = False
    ):
        # compute number of plans
        num_sentinel_plans = len(sentinel_plans)

        all_sample_execution_data, return_records = [], []
        results = list(map(lambda x:
                cls.run_sentinel_plan(*x),
                [(plan, idx, verbose) for idx, plan in enumerate(sentinel_plans)],
            )
        )
        # with ThreadPoolExecutor(max_workers=num_sentinel_plans) as executor:
        #     results = list(executor.map( lambda x:
        #             cls.run_sentinel_plan(*x),
        #             [(plan, idx, verbose) for idx, plan in enumerate(sentinel_plans)],
        #         )
        #     )
        print("Finished executing sentinel plans")
        # write out result dict and samples collected for each sentinel
        sentinel_records, sentinel_stats = zip(*results)
        for records, plan_stats, plan in zip(sentinel_records, sentinel_stats, sentinel_plans):
            # aggregate sentinel est. data
            sample_execution_data = cls.getSampleExecutionData(plan_stats)
            all_sample_execution_data.extend(sample_execution_data)

            # set return_records to be records from champion model
            champion_model_name = getChampionModelName()

            # find champion model plan records and add those to all_records
            if champion_model_name in plan.getPlanModelNames():
                return_records = records

        return all_sample_execution_data, return_records # TODO: make sure you capture cost of sentinel plans.

    @classmethod
    def execute_dag(cls, plan: PhysicalPlan, plan_stats: PlanStats):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards 
        """
        output_records = []
        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed

        for idx, operator in enumerate(plan.operators):
            op_id = operator.get_op_id()
            # TODO: Is it okay to have call return a list?
            if isinstance(operator, DataSourcePhysicalOperator):
                idx=0
                out_records, record_op_stats_lst = [], []
                num_samples = plan.num_samples if plan.num_samples else float("inf")

                ds = operator.datadir.getRegisteredDataset(plan.datasetIdentifier)
                for filename in sorted(os.listdir(ds.path)):
                    file_path = os.path.join(ds.path, filename)
                    if os.path.isfile(file_path):
                        if idx > num_samples:
                            break
                        record, record_op_stats = operator(file_path)
                        out_records.append(record)
                        record_op_stats_lst.append(record_op_stats)
                        idx += 1
            else:
                input_records = out_records
                out_records = []
                record_op_stats_lst = []
                for idx,input_record in enumerate(input_records):
                    if isinstance(operator, LimitScanOp) and idx == operator.limit:
                        break

                    records, record_op_stats = operator(input_record)
                    if records is None:
                        records = []
                        continue
                    elif type(records) != type([]):
                        records = [records]

                    for record in records:
                        if isinstance(operator, FilterOp):
                            if not record._passed_filter:
                                continue
                        out_records.append(record)
                    record_op_stats_lst.append(record_op_stats)

            # TODO code a nice __add__ function for OperatorStats and RecordOpStats
            for record_op_stats in record_op_stats_lst:
                x = plan_stats.operator_stats[op_id]
                x.record_op_stats_lst.append(record_op_stats)
                x.total_op_time += record_op_stats.op_time
                x.total_op_cost += record_op_stats.op_cost
                plan_stats.operator_stats[op_id] = x


        return out_records, plan_stats


    #TODO The dag style execution is not really working. I am implementing a per-records execution
    @classmethod
    def _todebug_execute_dag(cls, plan: PhysicalPlan, plan_stats: PlanStats):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards 
        """
        # initialize list of output records
        output_records = []
        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOperator)
        }

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        finished_executing = False
        while not finished_executing:
            more_source_records = False
            for idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()
                # TODO: Is it okay to have call return a list?
                if isinstance(operator, DataSourcePhysicalOperator):
                    records, record_op_stats_lst = operator()
                elif len(processing_queues[op_id]) > 0:
                    print(f"Processing operator {op_id} - queue length: {len(processing_queues[op_id])}")
                    input_record = processing_queues[op_id].pop(0)
                    records, record_op_stats = operator(input_record)
                    record_op_stats_lst = [record_op_stats]


                # update plan stats
                for record_op_stats in record_op_stats_lst:
                    # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                    x = plan_stats.operator_stats[op_id]
                    x.record_op_stats_lst.append(record_op_stats)
                    x.total_op_time += record_op_stats.op_time
                    x.total_op_cost += record_op_stats.op_cost
                    plan_stats.operator_stats[op_id] = x


                # get id for next physical operator (currently just next op in plan.operators)
                next_op_id = (
                    plan.operators[idx + 1].get_op_id()
                    if idx + 1 < len(plan.operators)
                    else None
                )
                if not isinstance(operator, DataSourcePhysicalOperator):
                    print(f"Now operator {op_id} - queue length: {len(processing_queues[op_id])}")

                if records is None:
                    continue
                elif type(records) != type([]):
                    records = [records]
                # update processing_queues or output_records
                for record in records:
                    if isinstance(operator, FilterOp):
                        if not record._passed_filter:
                            continue
                    if next_op_id is not None:
                        processing_queues[next_op_id].append(record)
                        print(f"For next operator {next_op_id} - queue length: {len(processing_queues[next_op_id])}")                    
                    else:
                        output_records.append(record)
                input("Press Enter to continue...")

            # update finished_executing based on whether all records have been processed
            still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
            finished_executing = not more_source_records and not still_processing
            print(f"Length of queue: {[len(queue) for queue in processing_queues.values()]}")
            print(f"More source records: {more_source_records}")
            input("Press Enter to continue...")
            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                finished_executing = (len(output_records) == operator.limit)

        return output_records, plan_stats

    @classmethod
    def execute(cls, plan: PhysicalPlan):
        """Initialize the stats and invoke _execute_dag() to execute the plan."""
        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id()) # TODO move into PhysicalPlan.__init__?
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here
        # NOTE: I am writing this execution helper function with the goal of supporting future
        #       physical plans that may have joins and/or other operations with multiple sources.
        #       Thus, the implementation is overkill for today's plans, but hopefully this will
        #       avoid the need for a lot of refactoring in the future.
        # execute the physical plan;
        output_records, plan_stats = cls.execute_dag(plan, plan_stats)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats

    # MR: per my limited understanding of staticmethods vs. classmethods, since this fcn.
    #     accepts cls as an argument I think it's supposed to be a classmethod?
    @classmethod
    def getSampleExecutionData(cls, plan_stats: PlanStats) -> List[SampleExecutionData]:
        """Compute and return all sample execution data collected by this plan so far."""
        # construct table of observation data from sample batch of processed records
        sample_execution_data, source_op_id = [], None
        for op_id, operator_stats in plan_stats.operator_stats.items():
            # append observation data for each record
            for record_op_stats in operator_stats.record_op_stats_lst:
                # compute minimal observation which is supported by all operators
                # TODO: one issue with this setup is that cache_scans of previously computed queries
                #       may not match w/these observations due to the diff. op_name
                observation_arguments = {
                    "record_uuid": record_op_stats.record_uuid,
                    "record_parent_uuid": record_op_stats.record_parent_uuid,
                    "op_id": op_id,
                    "op_name": record_op_stats.op_name,
                    "op_cost": record_op_stats.op_cost,
                    "source_op_id": source_op_id,
                    "op_time": record_op_stats.op_time,
                }
                if record_op_stats.op_details is not None:
                    observation_arguments.update({
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
                    ),}
                    )
                    answer = {}
                    # return key->value mapping for generated fields for induce
                    if "generated_fields" in record_op_stats.op_details:
                        for field in record_op_stats.op_details["generated_fields"]:
                            answer[field] = record_op_stats.record_state[field]
                    observation_arguments["answer"] = answer


                if record_op_stats.record_state is not None:
                    observation_arguments.update({
                    "total_input_tokens": (
                        record_op_stats.record_stats["total_input_tokens"]
                        if "total_input_tokens" in record_op_stats.record_state
                        else None
                    ),
                    "total_output_tokens": (
                        record_op_stats.record_stats["total_output_tokens"]
                        if "total_output_tokens" in record_op_stats.record_state
                        else None
                    ),
                    "total_input_cost": (
                        record_op_stats.record_stats["total_input_cost"]
                        if "total_input_cost" in record_op_stats.record_state
                        else None
                    ),
                    "total_output_cost": (
                        record_op_stats.record_stats["total_output_cost"]
                        if "total_output_cost" in record_op_stats.record_state
                        else None
                    ),
                    "answer": (record_op_stats.record_state["answer"] 
                        if "answer" in record_op_stats.record_state
                        else None),
                    "passed_filter": (
                        record_op_stats.record_state["_passed_filter"]
                        if "_passed_filter" in record_op_stats.record_state
                        else None
                    ),
                    })

                    # return T/F for filter
                    if "_passed_filter" in record_op_stats.record_state:
                        observation_arguments["answer"] = record_op_stats.record_state["_passed_filter"]

                observation = SampleExecutionData(**observation_arguments)
                # add observation to list of observations
                sample_execution_data.append(observation)

            # update source_op_id
            source_op_id = op_id

        return sample_execution_data