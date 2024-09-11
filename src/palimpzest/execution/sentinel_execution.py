from palimpzest.constants import OptimizationStrategy, PickOutputStrategy
from palimpzest.cost_model import CostModel
from palimpzest.dataclasses import ExecutionStats
from palimpzest.execution import (
    ExecutionEngine,
    PipelinedParallelSentinelPlanExecutor,
    PipelinedSingleThreadSentinelPlanExecutor,
    SequentialSingleThreadSentinelPlanExecutor,
)
from palimpzest.optimizer import Optimizer, SentinelPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set

from concurrent.futures import ThreadPoolExecutor

import time
import warnings

# TODO: we've removed the dataset_id; now we need this execution engine to:
#       - run on validation data (if present); otherwise run on first num_samples
#           - this should also be true for other execution engines
#       - have generate_sample_observations return records if there's no validation data (and only return observations otherwise)
#       - have execute_sentinel_plan mimic execute_plans; but handle copying the sentinel plan, and possibly passing a list of (record, col) --> plan
#           - then have it call the sentinel plan executor
class SentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute_plan() method.
    """
    def __init__(self, rank: int = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

        # check that user either provided validation data or set a finite number of samples
        self.using_validation_data = self.validation_data_source is not None
        assert self.using_validation_data or self.num_samples != float("inf"), "Must provide validation data or specify a finite number of samples"

        # TODO: relax this constraint once we've implemented an end-to-end working version of sentinel execution
        # check that the user provided at least 3 validation samples
        num_val_records = len(self.validation_data_source) if self.using_validation_data else self.num_samples
        assert num_val_records >= 3, "Number of validation examples (or samples) must be >= 3 to allow for low-rank approximation."

        # check that we have enough records for the specified rank
        if num_val_records < self.rank + 1:
            warnings.warn(f"Samples (n={num_val_records}) must be >= rank + 1 (r={self.rank}); Decreasing rank to {num_val_records - 1}")
            self.rank = num_val_records - 1

    def execute_sentinel_plan(self, plan: SentinelPlan, num_samples: int):
        """
        """
        # set plan_parallel_workers to be max workers and create one sentinel plan per worker
        plan_parallel_workers = min(self.max_workers, num_samples)
        plans = [plan.copy() for _ in range(plan_parallel_workers)]

        # split records across plans in contiguous (num_samples / plan_parallel_workers) sized chunks
        plan_record_indices = []
        for idx in range(plan_parallel_workers):
            start_idx = int(num_samples * (idx / plan_parallel_workers))
            end_idx = int(num_samples * ((idx + 1) / plan_parallel_workers))
            plan_record_indices.append((start_idx, end_idx))

        # TODO
        # if we're not using validation data; pass flag into execute_plan to
        # sample champion model for every record

        # divide records evenly among plans and execute
        # with ThreadPoolExecutor(max_workers=plan_parallel_workers) as executor:
            # results = list(executor.map(lambda x: self.execute_plan(**x),
            #         [{"plan": plan,
            #           "scan_start_idx": start_idx,
            #           "scan_end_idx": end_idx,
            #           "plan_workers": 1,
            #           }
            #           for plan, (start_idx, end_idx) in zip(plans, plan_record_indices)],
            #     )
            # )
        results = list(map(lambda x: self.execute_plan(**x),
                [{"plan": plan,
                  "scan_start_idx": start_idx,
                  "scan_end_idx": end_idx,
                  "plan_workers": 1}
                  for plan, (start_idx, end_idx) in zip(plans, plan_record_indices)],
            )
        )
        # split results into per-plan records and plan stats
        all_records, all_plan_stats = zip(*results)

        # process results to get sample execution data and sentinel plan stats
        all_sample_execution_data, return_records = [], None
        for records, plan_stats, plan in zip(all_records, all_plan_stats, plans):
            # aggregate sentinel est. data
            for operator_stats in plan_stats.operator_stats.values():
                all_sample_execution_data.extend(operator_stats.record_op_stats_lst)

            # if we're not using validation data; return results from the champion model or ensemble
            if not self.using_validation_data:
                return_records = records

        return all_sample_execution_data, return_records, all_plan_stats


    def generate_sample_observations(self, dataset: Set, policy: Policy):
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
            policy=policy,
            cost_model=CostModel(),
            no_cache=True,
            verbose=self.verbose,
            available_models=self.available_models,
            allow_bonded_query=self.allow_bonded_query,
            allow_conventional_query=self.allow_conventional_query,
            allow_code_synth=self.allow_code_synth,
            allow_token_reduction=self.allow_token_reduction,
            optimization_strategy=OptimizationStrategy.SENTINEL,
        )

        # if validation data is present, swap out the dataset's source with the validation source
        if self.using_validation_data:
            self.set_datasource(dataset, self.validation_data_source)

        # use optimizer to generate sentinel plans
        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        # run sentinel plan
        num_samples = (
            len(self.validation_data_source)
            if self.using_validation_data
            else min(self.num_samples, len(self.datasource))
        )

        execution_data, records, plan_stats = self.execute_sentinel_plan(sentinel_plan, num_samples)

        # if we ran on validation data, swap back to the original source
        if self.using_validation_data:
            self.set_datasource(dataset, self.datasource)
        
        # otherwise, advance the scan_start_idx
        else:
            self.scan_start_idx += num_samples

        return execution_data, records, plan_stats


    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # initialize the datasource
        self.init_datasource(dataset)

        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            self.clear_cached_responses_and_examples()

        # initialize execution data, records, and plan stats
        all_execution_data, all_records, all_plan_stats = [], [], []

        # NOTE: this checks if the entire computation is cached; it will re-run
        #       the sentinels even if the computation is partially cached
        # only run sentinels if there isn't a cached result already
        uid = dataset.universalIdentifier()
        run_sentinels = self.nocache or not self.datadir.hasCachedAnswer(uid)    
        if run_sentinels:
            all_execution_data, records, all_plan_stats = self.generate_sample_observations(dataset, policy)

            # if we did not use validation data; set the record outputs to be those from the champion model
            if not self.using_validation_data:
                all_records = records
        
            import json
            with open('tmp-execution-data.json', 'w') as f:
                all_execution_data = [rec_op_stats.to_json() for rec_op_stats in all_execution_data]
                json.dump(all_execution_data, f)
                exit(0)

        # TODO: pass in groundtruth answers to cost model as well (whether they are from validation source or champion sentinel)
        # construct the CostModel with any sample execution data we've gathered
        cost_model = CostModel(sample_execution_data=all_execution_data)

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


class PipelinedSingleThreadSentinelExecution(SentinelExecutionEngine, PipelinedSingleThreadSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, single-threaded fashion.
    """
    pass


class PipelinedParallelSentinelExecution(SentinelExecutionEngine, PipelinedParallelSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    pass
