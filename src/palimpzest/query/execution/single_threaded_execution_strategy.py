import logging
import time

from palimpzest.core.data.dataclasses import OperatorStats, PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan

logger = logging.getLogger(__name__)

class SequentialSingleThreadExecutionStrategy(ExecutionStrategy):
    """
    A single-threaded execution strategy that processes operators sequentially.
    
    This strategy processes all records through one operator completely before moving to the next operator
    in the execution plan. For example, if we have operators A -> B -> C and records [1,2,3]:
    1. First processes records [1,2,3] through operator A
    2. Then takes A's output and processes all of it through operator B
    3. Finally processes all of B's output through operator C
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {plan_workers} workers")
        logger.info(f"Plan Details: {plan}")

        plan_start_time = time.time()

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

        # initialize processing queues for each operation
        processing_queues = {op.get_op_id(): [] for op in plan.operators if not isinstance(op, ScanPhysicalOp)}

        # execute the plan one operator at a time
        for op_idx, operator in enumerate(plan.operators):
            logger.info(f"Processing operator {operator.op_name()} ({operator.get_op_id()})")

            op_id = operator.get_op_id()
            prev_op_id = plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
            next_op_id = plan.operators[op_idx + 1].get_op_id() if op_idx + 1 < len(plan.operators) else None

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

                    # update the current scan index
                    current_scan_idx += 1

                    # update whether to keep scanning source records
                    keep_scanning_source_records = current_scan_idx < datareader_len and len(records) < num_samples

            # aggregate operators accept all input records at once
            elif isinstance(operator, AggregateOp):
                record_set = operator(candidates=processing_queues[op_id])
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats

            # otherwise, process the records in the processing queue for this operator one at a time
            elif len(processing_queues[op_id]) > 0:
                for input_record in processing_queues[op_id]:
                    record_set = operator(input_record)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)

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

            logger.info(f"Finished processing operator {operator.op_name()} ({operator.get_op_id()}), and generated {len(records)} records")
            logger.debug(f"Records Stats: {record_op_stats}")

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

        logger.info(f"Completed execution of plan {plan.plan_id} in {time.time() - plan_start_time:.2f} seconds")
        logger.debug(f"Plan execution stats: {plan_stats}")

        return output_records, plan_stats


class PipelinedSingleThreadExecutionStrategy(ExecutionStrategy):
    """
    A single-threaded execution strategy that processes records through a pipeline of operators.
    
    This strategy implements a pipelined execution model where each record flows through
    the entire operator chain before the next record is processed.

    Example Flow:
    For operators A -> B -> C and records [1,2,3]:
    1. Record 1: A -> B -> C
    2. Record 2: A -> B -> C
    3. Record 3: A -> B -> C
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 1 if self.max_workers is None else self.max_workers

    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {plan_workers} workers")
        logger.info(f"Plan Details: {plan}")

        plan_start_time = time.time()

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

        # initialize processing queues for each operation
        processing_queues = {op.get_op_id(): [] for op in plan.operators if not isinstance(op, ScanPhysicalOp)}

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        finished_executing, keep_scanning_source_records = False, True
        while not finished_executing:
            for op_idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()
                logger.info(f"Processing operator {operator.op_name()} ({operator.get_op_id()}) over records_index={current_scan_idx}")
                
                prev_op_id = plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                next_op_id = plan.operators[op_idx + 1].get_op_id() if op_idx + 1 < len(plan.operators) else None

                # create empty lists for records and execution stats generated by executing this operator on its next input(s)
                records, record_op_stats = [], []

                # invoke scan operator(s) until we run out of source records or hit the num_samples limit
                if isinstance(operator, ScanPhysicalOp):
                    if keep_scanning_source_records:
                        # run ScanPhysicalOp on current scan index
                        record_set = operator(current_scan_idx)
                        records = record_set.data_records
                        record_op_stats = record_set.record_op_stats

                        # update number of source records scanned and the current index
                        source_records_scanned += len(records)
                        current_scan_idx += 1
                    else:
                        continue

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

                # otherwise, process the next record in the processing queue for this operator
                elif len(processing_queues[op_id]) > 0:
                    input_record = processing_queues[op_id].pop(0)
                    record_set = operator(input_record)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats

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

                logger.info(f"Finished processing operator {operator.op_name()} ({operator.get_op_id()}) over records_index={current_scan_idx}")

            # update finished_executing based on whether all records have been processed
            still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
            keep_scanning_source_records = current_scan_idx < datareader_len and source_records_scanned < num_samples
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
        logger.info(f"Completed execution of plan {plan.plan_id} in {time.time() - plan_start_time:.2f} seconds")
        logger.debug(f"Plan execution stats: (plan_str={plan_stats.plan_str}, plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
