import logging
import time

from palimpzest.core.data.dataclasses import ExecutionStats, OperatorStats, PlanStats
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.execution.parallel_execution_strategy import PipelinedParallelExecutionStrategy
from palimpzest.query.execution.single_threaded_execution_strategy import (
    PipelinedSingleThreadExecutionStrategy,
    SequentialSingleThreadExecutionStrategy,
)
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class NoSentinelQueryProcessor(QueryProcessor):
    """
    Specialized query processor that implements no sentinel strategy
    for coordinating optimization and execution.
    """

    # TODO: Consider to support dry_run.
    def execute(self) -> DataRecordCollection:
        logger.info("Executing NoSentinelQueryProcessor")
        execution_start_time = time.time()

        # if cache is False, make sure we do not re-use codegen examples
        if not self.cache:
            # self.clear_cached_examples()
            pass

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_with_strategy(self.dataset, self.policy, self.optimizer)

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

        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info("Done executing NoSentinelQueryProcessor")
        logger.debug(f"Result: {result}")
        return result


class NoSentinelSequentialSingleThreadProcessor(NoSentinelQueryProcessor, SequentialSingleThreadExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        SequentialSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose
        )
        self.progress_manager = None
        logger.info("Created NoSentinelSequentialSingleThreadProcessor")

    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Initialize the stats and execute the plan with progress reporting."""
        logger.info(f"Executing plan: {plan}")
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

        # get handle to scan operator and pre-compute its size
        source_operator = plan.operators[0]
        assert isinstance(source_operator, ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        datareader_len = len(source_operator.datareader)

        # Calculate total work units - each record needs to go through each operator
        total_ops = len(plan.operators)
        total_items = min(num_samples, datareader_len) if num_samples != float("inf") else datareader_len
        total_work_units = total_items * total_ops
        self.progress_manager.start(total_work_units)
        work_units_completed = 0

        # initialize processing queues for each operation
        processing_queues = {op.get_op_id(): [] for op in plan.operators if not isinstance(op, ScanPhysicalOp)}

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

                # invoke scan operator(s) until we run out of source records or hit the num_samples limit
                if isinstance(operator, ScanPhysicalOp):
                    keep_scanning_source_records = True
                    while keep_scanning_source_records:
                        # run ScanPhysicalOp on current scan index
                        record_set = operator(current_scan_idx)
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
                        keep_scanning_source_records = current_scan_idx < datareader_len and len(records) < num_samples

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
                if self.cache:
                    for record in records:
                        if getattr(record, "passed_operator", True):
                            # self.datadir.append_cache(operator.target_cache_id, record)
                            pass

                # update processing_queues or output_records
                for record in records:
                    if isinstance(operator, FilterOp) and not record.passed_operator:
                        continue
                    if next_op_id is not None:
                        processing_queues[next_op_id].append(record)
                    else:
                        output_records.append(record)

                # if we've filtered out all records, terminate early
                if next_op_id is not None and processing_queues[next_op_id] == []:
                    break

            # if caching was allowed, close the cache
            if self.cache:
                for _ in plan.operators:
                    # self.datadir.close_cache(operator.target_cache_id)
                    pass

            # finalize plan stats
            total_plan_time = time.time() - plan_start_time
            plan_stats.finalize(total_plan_time)

        finally:
            # Always finish progress tracking
            if self.progress_manager:
                self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_str={plan_stats.plan_str}, plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")
        return output_records, plan_stats


class NoSentinelPipelinedSingleThreadProcessor(NoSentinelQueryProcessor, PipelinedSingleThreadExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        PipelinedSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose
        )
        self.progress_manager = None
        logger.info("Created NoSentinelPipelinedSingleThreadProcessor")

    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Initialize the stats and execute the plan with progress reporting."""
        logger.info(f"Executing plan: {plan}")

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
        source_records_scanned = 0
        current_scan_idx = self.scan_start_idx

        # get handle to scan operator and pre-compute its size
        source_operator = plan.operators[0]
        assert isinstance(source_operator, ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        datareader_len = len(source_operator.datareader)

        # Calculate total work units - each record needs to go through each operator
        total_ops = len(plan.operators)
        total_items = min(num_samples, datareader_len) if num_samples != float("inf") else datareader_len
        total_work_units = total_items * total_ops
        self.progress_manager.start(total_work_units)
        work_units_completed = 0

        try:
            # initialize processing queues for each operation
            processing_queues = {op.get_op_id(): [] for op in plan.operators if not isinstance(op, ScanPhysicalOp)}

            # execute the plan until either:
            # 1. all records have been processed, or
            # 2. the final limit operation has completed
            finished_executing, keep_scanning_source_records = False, True
            while not finished_executing:
                for op_idx, operator in enumerate(plan.operators):
                    op_id = operator.get_op_id()
                    prev_op_id = plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                    next_op_id = plan.operators[op_idx + 1].get_op_id() if op_idx + 1 < len(plan.operators) else None

                    # Update progress with current operator info
                    op_name = operator.__class__.__name__
                    self.progress_manager.update(work_units_completed, f"Running {op_name} ({op_idx + 1}/{total_ops})")

                    # create empty lists for records and execution stats generated by executing this operator on its next input(s)
                    records, record_op_stats = [], []

                    # invoke scan operator(s) until we run out of source records or hit the num_samples limit
                    if isinstance(operator, ScanPhysicalOp):
                        if keep_scanning_source_records:
                            # run ScanPhysicalOp on current scan index
                            record_set = operator(current_scan_idx)
                            records = record_set.data_records
                            record_op_stats = record_set.record_op_stats

                            # Update progress for each processed record
                            work_units_completed += 1
                            self.progress_manager.update(
                                work_units_completed,
                                f"Scanning data source: {current_scan_idx + 1}/{total_items}"
                            )

                            # update number of source records scanned and the current index
                            source_records_scanned += len(records)
                            current_scan_idx += 1

                            # update whether to keep scanning source records
                            keep_scanning_source_records = current_scan_idx < datareader_len and source_records_scanned < num_samples

                    # only invoke aggregate operator(s) once there are no more source records and all
                    # upstream operators' processing queues are empty
                    elif isinstance(operator, AggregateOp):
                        upstream_ops_are_finished = True
                        for upstream_op_idx in range(op_idx):
                            # scan operators do not have processing queues
                            if isinstance(plan.operators[upstream_op_idx], ScanPhysicalOp):
                                continue

                            # check upstream ops which do have a processing queue
                            upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                            upstream_ops_are_finished = (
                                upstream_ops_are_finished and len(processing_queues[upstream_op_id]) == 0
                            )

                        if not keep_scanning_source_records and upstream_ops_are_finished:
                            record_set = operator(candidates=processing_queues[op_id])
                            records = record_set.data_records
                            record_op_stats = record_set.record_op_stats
                            processing_queues[op_id] = []

                            # Update progress for aggregate operation
                            work_units_completed += len(processing_queues[op_id])
                            self.progress_manager.update(
                                work_units_completed,
                                f"Aggregating {len(processing_queues[op_id])} records"
                            )

                    # otherwise, process the next record in the processing queue for this operator
                    elif len(processing_queues[op_id]) > 0:
                        input_record = processing_queues[op_id].pop(0)
                        record_set = operator(input_record)
                        records = record_set.data_records
                        record_op_stats = record_set.record_op_stats

                        # Update progress for processed record
                        work_units_completed += 1
                        self.progress_manager.update(
                            work_units_completed,
                            f"Processing record through {op_name}"
                        )

                    # if records were generated by this operator, process them
                    if len(records) > 0:
                        # update plan stats
                        plan_stats.operator_stats[op_id].add_record_op_stats(
                            record_op_stats,
                            source_op_id=prev_op_id,
                            plan_id=plan.plan_id,
                        )

                        # add records (which are not filtered) to the cache, if allowed
                        if self.cache:
                            for record in records:
                                if getattr(record, "passed_operator", True):
                                    # self.datadir.append_cache(operator.target_cache_id, record)
                                    pass

                        # update processing_queues or output_records
                        for record in records:
                            if isinstance(operator, FilterOp) and not record.passed_operator:
                                continue
                            if next_op_id is not None:
                                processing_queues[next_op_id].append(record)
                            else:
                                output_records.append(record)

                # update finished_executing based on whether all records have been processed
                still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
                finished_executing = not keep_scanning_source_records and not still_processing

                # update finished_executing based on limit
                if isinstance(operator, LimitScanOp):
                    finished_executing = len(output_records) == operator.limit

            # if caching was allowed, close the cache
            if self.cache:
                for _ in plan.operators:
                    # self.datadir.close_cache(operator.target_cache_id)
                    pass

            # finalize plan stats
            total_plan_time = time.time() - plan_start_time
            plan_stats.finalize(total_plan_time)

        finally:
            # Always finish progress tracking
            if self.progress_manager:
                self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_str={plan_stats.plan_str}, plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")
        return output_records, plan_stats


class NoSentinelPipelinedParallelProcessor(NoSentinelQueryProcessor, PipelinedParallelExecutionStrategy):
    """
    This class performs non-sample based execution while executing plans in a pipelined, parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        NoSentinelQueryProcessor.__init__(self, *args, **kwargs)
        PipelinedParallelExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            cache=self.cache,
            verbose=self.verbose
        )
        self.progress_manager = None
        logger.info("Created NoSentinelPipelinedParallelProcessor")

    # def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
    #     """Initialize the stats and execute the plan with progress reporting."""
    #     if self.verbose:
    #         print("----------------------")
    #         print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
    #         print(plan)
    #         print("---")

    #     plan_start_time = time.time()

    #     # Initialize progress manager
    #     self.progress_manager = create_progress_manager()

    #     # initialize plan stats and operator stats
    #     plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
    #     for op in plan.operators:
    #         op_id = op.get_op_id()
    #         op_name = op.op_name()
    #         op_details = {k: str(v) for k, v in op.get_id_params().items()}
    #         plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op_name, op_details=op_details)

    #     # initialize list of output records and intermediate variables
    #     output_records = []
    #     source_records_scanned = 0
    #     current_scan_idx = self.scan_start_idx

    #     # get handle to scan operator and pre-compute its size
    #     source_operator = plan.operators[0]
    #     assert isinstance(source_operator, ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
    #     datareader_len = len(source_operator.datareader)

    #     # Calculate total work units - each record needs to go through each operator
    #     total_ops = len(plan.operators)
    #     total_items = min(num_samples, datareader_len) if num_samples != float("inf") else datareader_len
    #     total_work_units = total_items * total_ops
    #     self.progress_manager.start(total_work_units)
    #     work_units_completed = 0

    #     try:
    #         with ThreadPoolExecutor(max_workers=plan_workers) as executor:
    #             # initialize processing queues and futures for each operation
    #             processing_queues = {op.get_op_id(): [] for op in plan.operators}
    #             futures = []

    #             # execute the plan until either:
    #             # 1. all records have been processed, or
    #             # 2. the final limit operation has completed
    #             finished_executing, keep_scanning_source_records = False, True
    #             last_work_units_completed = 0
    #             while not finished_executing:
    #                 # Process completed futures
    #                 done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)
    #                 futures = list(not_done_futures)

    #                 for future in done_futures:
    #                     record_set, operator, _ = future.result()
    #                     op_id = operator.get_op_id()
    #                     op_idx = next(i for i, op in enumerate(plan.operators) if op.get_op_id() == op_id)
    #                     next_op_id = plan.operators[op_idx + 1].get_op_id() if op_idx + 1 < len(plan.operators) else None

    #                     # Update progress for completed operation
    #                     work_units_completed += len(record_set.data_records)
    #                     if work_units_completed > last_work_units_completed:
    #                         self.progress_manager.update(
    #                             work_units_completed,
    #                             f"Completed {operator.__class__.__name__} on {len(record_set.data_records)} records"
    #                         )
    #                         last_work_units_completed = work_units_completed

    #                     # Process records
    #                     for record in record_set:
    #                         if isinstance(operator, FilterOp) and not record.passed_operator:
    #                             continue
    #                         if next_op_id is not None:
    #                             processing_queues[next_op_id].append(record)
    #                         else:
    #                             output_records.append(record)

    #                 # Submit new tasks
    #                 for _, operator in enumerate(plan.operators):
    #                     op_id = operator.get_op_id()
                        
    #                     if isinstance(operator, ScanPhysicalOp) and keep_scanning_source_records:
    #                         # Submit source operator task
    #                         futures.append(executor.submit(PhysicalOperator.execute_op_wrapper, operator, current_scan_idx))
    #                         current_scan_idx += 1
    #                         keep_scanning_source_records = current_scan_idx < datareader_len and source_records_scanned < num_samples
                        
    #                     elif len(processing_queues[op_id]) > 0:
    #                         # Submit task for next record in queue
    #                         input_record = processing_queues[op_id].pop(0)
    #                         futures.append(executor.submit(PhysicalOperator.execute_op_wrapper, operator, input_record))

    #                 # Check if we're done
    #                 still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
    #                 finished_executing = not keep_scanning_source_records and not still_processing and len(futures) == 0

    #         # if caching was allowed, close the cache
    #         if self.cache:
    #             for _ in plan.operators:
    #                 # self.datadir.close_cache(operator.target_cache_id)
    #                 pass

    #         # finalize plan stats
    #         total_plan_time = time.time() - plan_start_time
    #         plan_stats.finalize(total_plan_time)

    #     finally:
    #         # Always finish progress tracking
    #         if self.progress_manager:
    #             self.progress_manager.finish()

    #     return output_records, plan_stats
