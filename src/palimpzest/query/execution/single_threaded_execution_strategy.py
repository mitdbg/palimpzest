import logging

from palimpzest.core.data.dataclasses import PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.utils.progress import create_progress_manager

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
        self.max_workers = 1

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and the execute the plan."""
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert isinstance(plan.operators[0], ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize progress manager
        self.progress_manager = create_progress_manager(plan, self.num_samples, self.progress)

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues for each operation
        input_queues = self._create_input_queues(plan)

        # start the progress manager
        self.progress_manager.start()

        # execute the plan one operator at a time
        output_records = []
        for op_idx, operator in enumerate(plan.operators):
            # if we've filtered out all records, terminate early
            op_id = operator.get_op_id()
            num_inputs = len(input_queues[op_id])
            if num_inputs == 0:
                break

            # begin to process this operator
            records, record_op_stats = [], []
            logger.info(f"Processing operator {operator.op_name()} ({op_id})")

            # if this operator is an aggregate, process all the records in the input_queue
            if isinstance(operator, AggregateOp):
                record_set = operator(candidates=input_queues[op_id])
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats
                num_outputs = sum(record.passed_operator for record in records)

                # update the progress manager
                self.progress_manager.incr(op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # otherwise, process the records in the input queue for this operator one at a time
            else:
                for input_record in input_queues[op_id]:
                    record_set = operator(input_record)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)
                    num_outputs = sum(record.passed_operator for record in record_set.data_records)

                    # update the progress manager
                    self.progress_manager.incr(op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                    # finish early if this is a limit
                    if isinstance(operator, LimitScanOp) and len(records) == operator.limit:
                        break

            # update plan stats
            plan_stats.add_record_op_stats(record_op_stats)

            # add records to the cache
            self._add_records_to_cache(operator.target_cache_id, records)

            # update next input_queue (if it exists)
            output_records = [record for record in records if record.passed_operator]            
            if op_idx + 1 < len(plan.operators):
                next_op_id = plan.operators[op_idx + 1].get_op_id()
                input_queues[next_op_id] = output_records

            logger.info(f"Finished processing operator {operator.op_name()} ({operator.get_op_id()}), and generated {len(records)} records")

        # close the cache
        self._close_cache([op.target_cache_id for op in plan.operators])

        # finalize plan stats
        plan_stats.finish()

        # finish progress tracking
        self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

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
        self.max_workers = 1

    def _any_queue_not_empty(self, queues: dict[str, list]) -> bool:
        """Helper function to check if any queue is not empty."""
        return any(len(queue) > 0 for queue in queues.values())

    def _upstream_ops_finished(self, plan: PhysicalPlan, op_idx: int, input_queues: dict[str, list]) -> bool:
        """Helper function to check if all upstream operators have finished processing their inputs."""
        for upstream_op_idx in range(op_idx):
            upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
            if len(input_queues[upstream_op_id]) > 0:
                return False

        return True

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and the execute the plan."""
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert isinstance(plan.operators[0], ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize progress manager
        self.progress_manager = create_progress_manager(plan, self.num_samples, self.progress)

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues for each operation
        input_queues = self._create_input_queues(plan)

        # start the progress manager
        self.progress_manager.start()

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed (we break out of the loop if this happens)
        final_output_records = []
        while self._any_queue_not_empty(input_queues):
            for op_idx, operator in enumerate(plan.operators):
                # if this operator does not have enough inputs to execute, then skip it
                op_id = operator.get_op_id()
                num_inputs = len(input_queues[op_id])
                agg_op_not_ready = isinstance(operator, AggregateOp) and not self._upstream_ops_finished(plan, op_idx, input_queues)
                if num_inputs == 0 or agg_op_not_ready:
                    continue

                # create empty lists for records and execution stats generated by executing this operator on its next input(s)
                records, record_op_stats = [], []

                # if the next operator is an aggregate, process all the records in the input_queue
                if isinstance(operator, AggregateOp):
                    input_records = [input_queues[op_id].pop(0) for _ in range(num_inputs)]
                    record_set = operator(candidates=input_records)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    num_outputs = sum(record.passed_operator for record in records)

                    # update the progress manager
                    self.progress_manager.incr(op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                # otherwise, process the next record in the input queue for this operator
                else:
                    input_record = input_queues[op_id].pop(0)
                    record_set = operator(input_record)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    num_outputs = sum(record.passed_operator for record in records)

                    # update the progress manager
                    self.progress_manager.incr(op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                # update plan stats
                plan_stats.add_record_op_stats(record_op_stats)

                # add records to the cache
                self._add_records_to_cache(operator.target_cache_id, records)

                # update next input_queue or final_output_records
                output_records = [record for record in records if record.passed_operator]            
                if op_idx + 1 < len(plan.operators):
                    next_op_id = plan.operators[op_idx + 1].get_op_id()
                    input_queues[next_op_id].extend(output_records)
                else:
                    final_output_records.extend(output_records)

                logger.info(f"Finished processing operator {operator.op_name()} ({operator.get_op_id()}) on {num_inputs} records")

            # break out of loop if the final operator is a LimitScanOp and we've reached its limit
            if isinstance(plan.operators[-1], LimitScanOp) and len(final_output_records) == plan.operators[-1].limit:
                break

        # close the cache
        self._close_cache([op.target_cache_id for op in plan.operators])

        # finalize plan stats
        plan_stats.finish()

        # finish progress tracking
        self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return final_output_records, plan_stats
