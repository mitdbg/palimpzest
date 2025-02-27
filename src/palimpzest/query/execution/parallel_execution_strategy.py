import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import PlanStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class ParallelExecutionStrategy(ExecutionStrategy):
    """
    A parallel execution strategy that processes data through a pipeline of operators using thread-based parallelism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = (
            self._get_parallel_max_workers()
            if self.max_workers is None
            else self.max_workers
        )

    def _get_parallel_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return max(int(0.8 * multiprocessing.cpu_count()), 1)

    def _any_queue_not_empty(self, queues: dict[str, list]) -> bool:
        """Helper function to check if any queue is not empty."""
        return any(len(queue) > 0 for queue in queues.values())

    def _upstream_ops_finished(self, plan: PhysicalPlan, op_idx: int, input_queues: dict[str, list], future_queues: dict[str, list]) -> bool:
        """Helper function to check if all upstream operators have finished processing their inputs."""
        for upstream_op_idx in range(op_idx):
            upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
            if len(input_queues[upstream_op_id]) > 0 or len(future_queues[upstream_op_id]) > 0:
                return False

        return True

    def _process_future_results(self, operator: PhysicalOperator, future_queues: dict[str, list], plan_stats: PlanStats) -> list[DataRecord]:
        """
        Helper function which takes an operator, the future queues, and plan stats, and performs
        the updates to plan stats and progress manager before returning the results from the finished futures.
        """
        # get the op_id for the operator
        op_id = operator.get_op_id()

        # this function is called when the future queue is not empty
        # and the executor is not busy processing other futures
        done_futures, not_done_futures = wait(future_queues[op_id], timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

        # add the unfinished futures back to the previous op's future queue
        future_queues[op_id] = list(not_done_futures)

        # add the finished futures to the input queue for this operator
        output_records = []
        for future in done_futures:
            record_set: DataRecordSet = future.result()
            records = record_set.data_records
            record_op_stats = record_set.record_op_stats
            num_outputs = sum(record.passed_operator for record in records)

            # update the progress manager
            self.progress_manager.incr(op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # update plan stats
            plan_stats.add_record_op_stats(record_op_stats)

            # add records to the cache
            self._add_records_to_cache(operator.target_cache_id, records)

            # add records which aren't filtered to the output records
            output_records.extend([record for record in records if record.passed_operator])
        
        return output_records

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

        # initialize input queues and future queues for each operation
        input_queues = self._create_input_queues(plan)
        future_queues = {op.get_op_id(): [] for op in plan.operators}

        # start the progress manager
        self.progress_manager.start()

        # process all of the input records using a thread pool
        output_records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            logger.debug(f"Created thread pool with {self.max_workers} workers")

            # execute the plan until either:
            # 1. all records have been processed, or
            # 2. the final limit operation has completed (we break out of the loop if this happens)
            final_op = plan.operators[-1]
            while self._any_queue_not_empty(input_queues) or self._any_queue_not_empty(future_queues):
                for op_idx, operator in enumerate(plan.operators):
                    op_id = operator.get_op_id()

                    # get any finished futures from the previous operator and add them to the input queue for this operator
                    if not isinstance(operator, ScanPhysicalOp):
                        prev_operator = plan.operators[op_idx - 1]
                        records = self._process_future_results(prev_operator, future_queues, plan_stats)
                        input_queues[op_id].extend(records)

                    # for the final operator, add any finished futures to the output_records
                    if operator.get_op_id() == final_op.get_op_id():
                        records = self._process_future_results(operator, future_queues, plan_stats)
                        output_records.extend(records)

                    # if this operator does not have enough inputs to execute, then skip it
                    num_inputs = len(input_queues[op_id])
                    agg_op_not_ready = isinstance(operator, AggregateOp) and not self._upstream_ops_finished(plan, op_idx, input_queues, future_queues)
                    if num_inputs == 0 or agg_op_not_ready:
                        continue

                    # if this operator is an aggregate, process all the records in the input queue
                    if isinstance(operator, AggregateOp):
                        input_records = [input_queues[op_id].pop(0) for _ in range(num_inputs)]
                        future = executor.submit(operator, input_records)
                        future_queues[op_id].append(future)

                    else:
                        input_record = input_queues[op_id].pop(0)
                        future = executor.submit(operator, input_record)
                        future_queues[op_id].append(future)

                # break out of loop if the final operator is a LimitScanOp and we've reached its limit
                if isinstance(final_op, LimitScanOp) and len(output_records) == final_op.limit:
                    break

        # close the cache
        self._close_cache([op.target_cache_id for op in plan.operators])

        # finalize plan stats
        plan_stats.finish()

        # finish progress tracking
        self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
