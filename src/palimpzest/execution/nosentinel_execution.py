import time
from palimpzest.constants import PlanType
from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import DataRecord
from palimpzest.operators import AggregateOp, DataSourcePhysicalOp, LimitScanOp, MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp
from palimpzest.planner import LogicalPlanner, PhysicalPlanner, PhysicalPlan
from palimpzest.policy import Policy
from .cost_estimator import CostEstimator
from .execution import SimpleExecution
from palimpzest.sets import Set

from palimpzest.dataclasses import OperatorStats, PlanStats

from typing import Optional

import os
import shutil


class NoSentinelExecution(SimpleExecution):
    """ This class is a dummy execution engine that can be used for testing purposes. 
    It does not include sentinel plans or all that jazz."""

    def __init__(self,
            *args, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)

    def execute(self,
        dataset: Set,
        policy: Policy,
    ):
        # Always delete cache
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)
        cache = self.datadir.getCacheService()
        cache.rmCache()

        self.set_source_dataset_id(dataset)

        # NOTE: this checks if the entire computation is cached; it will re-run
        #       the sentinels even if the computation is partially cached
        # only run sentinels if there isn't a cached result already
        uid = dataset.universalIdentifier()
        logical_planner = LogicalPlanner(no_cache=True)
        physical_planner = PhysicalPlanner(
            num_samples=self.num_samples,
            scan_start_idx=0,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_model_selection=self.allow_model_selection,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            useParallelOps=self.useParallelOps,
        )

        # enumerate all possible physical plans
        all_physical_plans = []
        for logical_plan in logical_planner.generate_plans(dataset):
            for physical_plan in physical_planner.generate_plans(logical_plan):
                all_physical_plans.append(physical_plan)

        # construct the CostEstimator with any sample execution data we've gathered
        cost_estimator = CostEstimator(source_dataset_id=self.source_dataset_id)

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
        plan = policy.choose(plans)
        new_records, stats = self.execute_plan(plan, plan_type=PlanType.FINAL) # TODO: Still WIP
        all_records = new_records

        return all_records, plan, stats

    # TODO The dag style execution is not really working. I am implementing a per-records execution
    def execute_sequential(self, plan: PhysicalPlan, plan_stats: PlanStats):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards the future.
        """
        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0
        datasource_len = 0
        current_scan_idx = self.scan_start_idx

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }
        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        finished_executing, keep_scanning_source_records = False, True
        while not finished_executing:
            for op_idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()

                prev_op_id = (
                    plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                )
                next_op_id = (
                    plan.operators[op_idx + 1].get_op_id()
                    if op_idx + 1 < len(plan.operators)
                    else None
                )

                # TODO: if self.useParallelOps is True; execute each operator with parallelism
                # invoke datasource operator(s) until we run out of source records
                if isinstance(operator, DataSourcePhysicalOp) and keep_scanning_source_records:
                    # get handle to DataSource and pre-compute its size
                    datasource = (
                        self.datadir.getRegisteredDataset(self.source_dataset_id)
                        if isinstance(operator, MarshalAndScanDataOp)
                        else self.datadir.getCachedResult(operator.cachedDataIdentifier)
                    )
                    datasource_len = len(datasource)

                    # construct input DataRecord for DataSourcePhysicalOp
                    candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                    candidate.idx = current_scan_idx
                    candidate.get_item_fn = datasource.getItem
                    candidate.cardinality = datasource.cardinality

                    # run DataSourcePhysicalOp on record
                    records, record_op_stats_lst = operator(candidate)

                    # update number of source records scanned and the current index
                    source_records_scanned += len(records)
                    current_scan_idx += 1

                # only invoke aggregate operator(s) once there are no more source records and all
                # upstream operators' processing queues are empty
                elif isinstance(operator, AggregateOp):
                    upstream_queues_are_empty = True
                    for upstream_op_idx in range(op_idx):
                        upstream_queues_are_empty = (
                            upstream_queues_are_empty
                            and len(processing_queues[upstream_op_idx]) == 0
                        )
                    if not keep_scanning_source_records and upstream_queues_are_empty:
                        records, record_op_stats_lst = operator(candidates=processing_queues[op_idx])

                # otherwise, process the next record in the processing queue for this operator
                elif len(processing_queues[op_id]) > 0:
                    print(f"Processing operator {op_id} - queue length: {len(processing_queues[op_id])}")
                    input_record = processing_queues[op_id].pop(0)
                    records, record_op_stats_lst = operator(input_record)

                # update plan stats
                op_stats = plan_stats.operator_stats[op_id]
                for record_op_stats in record_op_stats_lst:
                    # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                    record_op_stats.source_op_id = prev_op_id
                    op_stats.record_op_stats_lst.append(record_op_stats)
                    op_stats.total_op_time += record_op_stats.time_per_record
                    op_stats.total_op_cost += record_op_stats.cost_per_record

                plan_stats.operator_stats[op_id] = op_stats

                # TODO some operator is not returning a singleton list
                if type(records) != type([]):
                    records = [records]

                # TODO: manage the cache here

                # update processing_queues or output_records
                for record in records:
                    if isinstance(operator, FilterOp):
                        if not record._passed_filter:
                            continue
                    if next_op_id is not None:
                        processing_queues[next_op_id].append(record)
                    else:
                        output_records.append(record)


            # update finished_executing based on whether all records have been processed
            still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
            keep_scanning_source_records = (
                current_scan_idx < datasource_len
            )
            finished_executing = not keep_scanning_source_records and not still_processing

            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                finished_executing = (len(output_records) == operator.limit)

        return output_records, plan_stats

    # NOTE: Adding a few optional arguments for printing, etc.
    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None):
        """Initialize the stats and invoke execute_dag() to execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"{plan_type.value} {str(plan_idx)}:")
            plan.printPlan()
            print("---")

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
        output_records, plan_stats = self.execute_sequential(plan, plan_stats)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats

