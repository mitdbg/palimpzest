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
from .execution import ExecutionEngine
from palimpzest.sets import Set

from palimpzest.dataclasses import OperatorStats, PlanStats

from typing import Optional

import os
import shutil


class StreamingSequentialExecution(ExecutionEngine):
    """ This class can be used for a streaming, record-based execution.
    Results are returned as an iterable that can be consumed by the caller."""

    def __init__(self,
            *args, **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.last_record = False
        self.current_scan_idx = 0
        self.plan = None
        self.plan_stats = None


    def generate_plan(self, dataset: Set, policy: Policy):
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)
        cache = self.datadir.getCacheService()
        cache.rmCache()

        self.set_source_dataset_id(dataset)
        start_time = time.time()
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

        plans = physical_planner.deduplicate_plans(all_physical_plans)
        final_plans = physical_planner.select_pareto_optimal_plans(plans)

        # for experimental evaluation, we may want to include baseline plans
        if self.include_baselines:
            final_plans = physical_planner.add_baseline_plans(final_plans)

        if self.min_plans is not None and len(final_plans) < self.min_plans:
            final_plans = physical_planner.add_plans_closest_to_frontier(final_plans, plans, self.min_plans)

        # choose best plan and execute it
        self.plan = policy.choose(plans)

        # TODO move into PhysicalPlan.__init__?
        self.plan_stats = PlanStats(plan_id=self.plan.plan_id())
        print(f"Time for planning: {time.time() - start_time}")
        for op_idx, op in enumerate(self.plan.operators):
            op_id = op.get_op_id()
            self.plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) 

        return self.plan
        # if self.verbose:
            # print("----------------------")
            # print(f"{plan_type.value} {str(plan_idx)}:")
            # plan.printPlan()
            # print("---")

    def execute(self,
        dataset: Set,
        policy: Policy,
    ):

        start_time = time.time()
        # Always delete cache
        if not self.current_scan_idx:
            self.generate_plan(self, dataset, policy)

        while not self.last_record:
            records, stats = self.execute_stream()
            yield records, self.plan, stats

        if self.last_record:
            # finalize plan stats
            total_plan_time = time.time() - start_time
            self.plan_stats.finalize(total_plan_time)
        yield records, self.plan, stats

    def execute_stream(self):
        """
        Helper function which executes the physical plan. This function is overly complex for today's
        plans which are simple cascades -- but is designed with an eye towards the future.
        """
        # initialize list of output records and intermediate variables
        plan = self.plan
        plan_stats = self.plan_stats
        records = []

        for op_idx, operator in enumerate(plan.operators):

            op_id = operator.get_op_id()

            prev_op_id = (
                plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
            )

            if isinstance(operator, DataSourcePhysicalOp):
                datasource = (
                    self.datadir.getRegisteredDataset(self.source_dataset_id)
                    if isinstance(operator, MarshalAndScanDataOp)
                    else self.datadir.getCachedResult(operator.cachedDataIdentifier)
                )
                datasource_len = len(datasource)

                # construct input DataRecord for DataSourcePhysicalOp
                candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=self.current_scan_idx)
                candidate.idx = self.current_scan_idx
                candidate.get_item_fn = datasource.getItem
                candidate.cardinality = datasource.cardinality
                records, record_op_stats_lst = operator(candidate)
                self.current_scan_idx += 1

            # only invoke aggregate operator(s) once there are no more source records and all
            # upstream operators' processing queues are empty
            elif isinstance(operator, AggregateOp):
                records, record_op_stats_lst = operator(candidates=records)

            # otherwise, process the next record in the processing queue for this operator
            else:
                records, record_op_stats_lst = operator(records[0])

            # update plan stats
            op_stats = plan_stats.operator_stats[op_id]
            for record_op_stats in record_op_stats_lst:
                # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                record_op_stats.source_op_id = prev_op_id
                op_stats.record_op_stats_lst.append(record_op_stats)
                op_stats.total_op_time += record_op_stats.time_per_record
                op_stats.total_op_cost += record_op_stats.cost_per_record

            plan_stats.operator_stats[op_id] = op_stats
            self.last_record = self.current_scan_idx == datasource_len

            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                self.last_record = (self.current_scan_idx == operator.limit)

            if isinstance(operator, FilterOp):
                # delete all records that did not pass the filter
                records = [r for r in records if r._passed_filter]
                if not records:
                    break
            
        print("Iteration number: ", self.current_scan_idx, "Last record: ", self.last_record)

        return records, plan_stats