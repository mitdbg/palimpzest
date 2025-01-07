import time

from palimpzest.constants import OptimizationStrategy
from palimpzest.core.data.dataclasses import ExecutionStats, OperatorStats, PlanStats
from palimpzest.core.lib.schemas import SourceRecord
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.execution.execution_engine import ExecutionEngine
from palimpzest.query.execution.plan_executors.parallel_plan_execution import (
    PipelinedParallelPlanExecutor,
)
from palimpzest.query.execution.plan_executors.single_threaded_plan_execution import (
    PipelinedSingleThreadPlanExecutor,
    SequentialSingleThreadPlanExecutor,
)
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.datasource import DataSourcePhysicalOp, MarshalAndScanDataOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set
from palimpzest.utils.progress import ProgressManager, create_progress_manager


class NoSentinelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the execute_plan() method.
    """

    def execute(self, dataset: Set, policy: Policy):
        execution_start_time = time.time()

        # if nocache is True, make sure we do not re-use DSPy examples or codegen examples
        if self.nocache:
            self.clear_cached_responses_and_examples()

        # construct the CostModel
        cost_model = CostModel()

        # initialize the optimizer
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
        records, plan_stats = [], []
        if self.optimization_strategy == OptimizationStrategy.CONFIDENCE_INTERVAL:
            records, plan_stats = self.execute_confidence_interval_strategy(dataset, policy, optimizer)
        
        else:
            records, plan_stats = self.execute_strategy(dataset, policy, optimizer)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(
                list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))
            ),
            plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
        )

        return records, execution_stats


class NoSentinelSequentialSingleThreadExecution(NoSentinelExecutionEngine, SequentialSingleThreadPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        SequentialSingleThreadPlanExecutor.__init__(self, *args, **kwargs)
        self.progress_manager = None

    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Initialize the stats and execute the plan with progress reporting."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # Initialize progress manager
        self.progress_manager = create_progress_manager()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op in plan.operators:
            op_id = op.get_op_id()
            op_name = op.op_name()
            op_details = {k: str(v) for k, v in op.get_id_params().items()}
            plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op_name, op_details=op_details)

        # initialize list of output records and intermediate variables
        output_records = []
        current_scan_idx = self.scan_start_idx

        # get handle to DataSource and pre-compute its size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.get_registered_dataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.get_cached_result(source_operator.dataset_id)
        )
        datasource_len = len(datasource)

        # Calculate total work units - each record needs to go through each operator
        total_ops = len(plan.operators)
        total_items = min(num_samples, datasource_len) if num_samples != float("inf") else datasource_len
        total_work_units = total_items * total_ops
        self.progress_manager.start(total_work_units)
        work_units_completed = 0

        # initialize processing queues for each operation
        processing_queues = {op.get_op_id(): [] for op in plan.operators if not isinstance(op, DataSourcePhysicalOp)}

        try:
            # execute the plan one operator at a time
            for op_idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()
                prev_op_id = plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                next_op_id = plan.operators[op_idx + 1].get_op_id() if op_idx + 1 < len(plan.operators) else None

                # Update progress to show which operator is currently running
                op_name = operator.__class__.__name__
                self.progress_manager.update(work_units_completed, f"Running {op_name} ({op_idx + 1}/{total_ops})")

                # initialize output records and record_op_stats for this operator
                records, record_op_stats = [], []

                # invoke datasource operator(s) until we run out of source records or hit the num_samples limit
                if isinstance(operator, DataSourcePhysicalOp):
                    keep_scanning_source_records = True
                    while keep_scanning_source_records:
                        # construct input DataRecord for DataSourcePhysicalOp
                        # NOTE: this DataRecord will be discarded and replaced by the scan_operator;
                        #       it is simply a vessel to inform the scan_operator which record to fetch
                        candidate = DataRecord(schema=SourceRecord, source_id=current_scan_idx)
                        candidate.idx = current_scan_idx
                        candidate.get_item_fn = datasource.get_item

                        # run DataSourcePhysicalOp on record
                        record_set = operator(candidate)
                        records.extend(record_set.data_records)
                        record_op_stats.extend(record_set.record_op_stats)

                        # Update progress for each processed record in data source
                        work_units_completed += 1
                        self.progress_manager.update(
                            work_units_completed, 
                            f"Scanning data source: {current_scan_idx + 1}/{total_items}"
                        )

                        # update the current scan index
                        current_scan_idx += 1

                        # update whether to keep scanning source records
                        keep_scanning_source_records = current_scan_idx < datasource_len and len(records) < num_samples

                # aggregate operators accept all input records at once
                elif isinstance(operator, AggregateOp):
                    record_set = operator(candidates=processing_queues[op_id])
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    
                    # Update progress for aggregate operation - count all records being aggregated
                    work_units_completed += len(processing_queues[op_id])
                    self.progress_manager.update(
                        work_units_completed,
                        f"Aggregating {len(processing_queues[op_id])} records"
                    )

                # otherwise, process the records in the processing queue for this operator one at a time
                elif len(processing_queues[op_id]) > 0:
                    queue_size = len(processing_queues[op_id])
                    for idx, input_record in enumerate(processing_queues[op_id]):
                        record_set = operator(input_record)
                        records.extend(record_set.data_records)
                        record_op_stats.extend(record_set.record_op_stats)

                        # Update progress for each processed record in the queue
                        work_units_completed += 1
                        self.progress_manager.update(
                            work_units_completed,
                            f"Processing records: {idx + 1}/{queue_size}"
                        )

                        if isinstance(operator, LimitScanOp) and len(records) == operator.limit:
                            break

                # update plan stats
                plan_stats.operator_stats[op_id].add_record_op_stats(
                    record_op_stats,
                    source_op_id=prev_op_id,
                    plan_id=plan.plan_id,
                )

                # add records (which are not filtered) to the cache, if allowed
                if not self.nocache:
                    for record in records:
                        if getattr(record, "_passed_operator", True):
                            self.datadir.append_cache(operator.target_cache_id, record)

                # update processing_queues or output_records
                for record in records:
                    if isinstance(operator, FilterOp) and not record._passed_operator:
                        continue
                    if next_op_id is not None:
                        processing_queues[next_op_id].append(record)
                    else:
                        output_records.append(record)

                # if we've filtered out all records, terminate early
                if next_op_id is not None and processing_queues[next_op_id] == []:
                    break

            # if caching was allowed, close the cache
            if not self.nocache:
                for operator in plan.operators:
                    self.datadir.close_cache(operator.target_cache_id)

            # finalize plan stats
            total_plan_time = time.time() - plan_start_time
            plan_stats.finalize(total_plan_time)

        finally:
            # Always finish progress tracking
            if self.progress_manager:
                self.progress_manager.finish()

        return output_records, plan_stats


class NoSentinelPipelinedSingleThreadExecution(NoSentinelExecutionEngine, PipelinedSingleThreadPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a pipelined, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        PipelinedSingleThreadPlanExecutor.__init__(self, *args, **kwargs)


class NoSentinelPipelinedParallelExecution(NoSentinelExecutionEngine, PipelinedParallelPlanExecutor):
    """
    This class performs non-sample based execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelExecutionEngine.__init__(self, *args, **kwargs)
        PipelinedParallelPlanExecutor.__init__(self, *args, **kwargs)
