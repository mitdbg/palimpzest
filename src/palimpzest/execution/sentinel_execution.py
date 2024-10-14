from palimpzest.constants import OptimizationStrategy
from palimpzest.dataclasses import ExecutionStats
from palimpzest.elements import DataRecordSet
from palimpzest.execution import (
    ExecutionEngine,
    SequentialSingleThreadSentinelPlanExecutor,
    SequentialParallelSentinelPlanExecutor
)
from palimpzest.optimizer import (
    CostModel,
    MatrixCompletionCostModel,
    Optimizer,
    SentinelPlan,
)
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
    def __init__(self, rank: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

        # check that user either provided validation data or set a finite number of samples
        assert self.using_validation_data or self.num_samples != float("inf"), "Must provide validation data or specify a finite number of samples"

        # TODO: relax this constraint once we've implemented an end-to-end working version of sentinel execution
        # check that the user provided at least 3 validation samples
        num_val_records = self.datasource.getValLength() if self.using_validation_data else self.num_samples
        assert num_val_records >= 3, "Number of validation examples (or samples) must be >= 3 to allow for low-rank approximation."

        # check that we have enough records for the specified rank
        if num_val_records < self.rank + 1:
            print(f"Samples (n={num_val_records}) must be >= rank + 1 (r={self.rank}); Decreasing rank to {num_val_records - 1}")
            self.rank = num_val_records - 1

    def run_sentinel_plan(self, plan: SentinelPlan, num_samples: int):
        """
        """
        # set plan_parallel_workers to be max workers and create one sentinel plan per worker
        plan_parallel_workers = min(self.max_workers, num_samples)

        # execute sentinel plan on the specified number of samples with as much parallelism as possible
        results = self.execute_sentinel_plan(plan, num_samples, plan_parallel_workers)

        # split results into per-plan records and plan stats
        sample_execution_data, champion_outputs, plan_stats = results

        return sample_execution_data, champion_outputs, plan_stats


    def generate_sample_observations(self, sentinel_plan: SentinelPlan):
        """
        This function is responsible for generating sample observation data which can be
        consumed by the CostModel. For each physical optimization and each operator, our
        goal is to capture `rank + 1` samples per optimization, where `rank` is the presumed
        low-rank of the observation matrix.

        To accomplish this, we construct a special sentinel plan using the Optimizer which is
        capable of executing any valid physical implementation of a Filter or Convert operator
        on each record.
        """
        # run sentinel plan
        num_samples = (
            self.datasource.getValLength()
            if self.using_validation_data
            else min(self.num_samples, len(self.datasource))
        )
        execution_data, champion_outputs, plan_stats = self.run_sentinel_plan(sentinel_plan, num_samples)

        # if we're not using validation data, advance the scan_start_idx
        if not self.using_validation_data:
            self.scan_start_idx += num_samples

        # if we're using validation data, get the set of expected output records
        expected_outputs, field_to_metric_fn = None, None
        if self.using_validation_data:
            field_to_metric_fn = self.datasource.getFieldToMetricFn()
            expected_outputs = {}
            for idx in range(self.datasource.getValLength()):
                data_records = self.datasource.getItem(idx, val=True, include_label=True)
                if type(data_records) != type([]):
                    data_records = [data_records]
                record_set = DataRecordSet(data_records, None)
                expected_outputs[record_set.source_id] = record_set

        return execution_data, champion_outputs, expected_outputs, field_to_metric_fn, plan_stats


    def create_sentinel_plan(self, dataset: Set, policy: Policy) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
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

        # # if validation data is present, swap out the dataset's source with the validation source
        # if self.using_validation_data:
        #     self.set_datasource(dataset, self.validation_data_source)

        # use optimizer to generate sentinel plans
        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        # # swap back to the original datasource if necessary
        # if self.using_validation_data:
        #     self.set_datasource(dataset, self.datasource)

        return sentinel_plan


    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            self.clear_cached_responses_and_examples()

        # create sentinel plan
        sentinel_plan = self.create_sentinel_plan(dataset, policy)

        # generate sample execution data
        all_execution_data, champion_outputs, expected_outputs, field_to_metric_fn, plan_stats = self.generate_sample_observations(sentinel_plan)

        # if we did not use validation data; set the record outputs to be those from the champion model or ensemble
        all_records = []
        if not self.using_validation_data:
            final_op_set = sentinel_plan.operator_sets[-1]
            final_op_set_id = SentinelPlan.compute_op_set_id(final_op_set)
            all_records = champion_outputs[final_op_set_id]

        # put sentinel plan execution stats into list
        all_plan_stats = [plan_stats]

        # construct the CostModel with any sample execution data we've gathered
        cost_model = MatrixCompletionCostModel(
            sentinel_plan=sentinel_plan,
            rank=self.rank,
            execution_data=all_execution_data,
            champion_outputs=champion_outputs,
            expected_outputs=expected_outputs,
            field_to_metric_fn=field_to_metric_fn,
            verbose=self.verbose,
        )

        # TODO: remove after SIGMOD
        import os
        if os.environ['LOG_MATRICES'].lower() == "true":
            exit(1)

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
        total_optimization_time = time.time() - execution_start_time

        # execute plan(s) according to the optimization strategy
        if self.optimization_strategy == OptimizationStrategy.OPTIMAL:
            records, plan_stats = self.execute_optimal_strategy(dataset, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        elif self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            records, plan_stats = self.execute_confidence_interval_strategy(dataset, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        elif self.optimization_strategy == OptimizationStrategy.NONE:
            records, plan_stats = self.execute_naive_strategy(dataset, optimizer)
            all_records.extend(records)
            all_plan_stats.extend(plan_stats)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(all_plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_optimization_time=total_optimization_time,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))),
            plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
        )

        return all_records, execution_stats


class SequentialSingleThreadSentinelExecution(SentinelExecutionEngine, SequentialSingleThreadSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        SentinelExecutionEngine.__init__(self, *args, **kwargs)
        SequentialSingleThreadSentinelPlanExecutor.__init__(self, *args, **kwargs)


class SequentialParallelSentinelExecution(SentinelExecutionEngine, SequentialParallelSentinelPlanExecutor):
    """
    This class performs sentinel execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        SentinelExecutionEngine.__init__(self, *args, **kwargs)
        SequentialParallelSentinelPlanExecutor.__init__(self, *args, **kwargs)
