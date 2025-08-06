import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, wait

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
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

    def _any_queue_not_empty(self, queues: dict[str, list] | dict[str, dict[str, list]]) -> bool:
        """Helper function to check if any queue is not empty."""
        for _, value in queues.items():
            if isinstance(value, dict):
                if any(len(subqueue) > 0 for subqueue in value.values()):
                    return True
            elif len(value) > 0:
                return True
        return False

    def _upstream_ops_finished(self, plan: PhysicalPlan, topo_idx: int, operator: PhysicalOperator, input_queues: dict[str, dict[str, list]], future_queues: dict[str, list]) -> bool:
        """Helper function to check if agg / join operator is ready to process its inputs."""
        # for agg / join operator, we can only process it when all upstream operators have finished processing their inputs
        upstream_unique_full_op_ids = plan.get_upstream_unique_full_op_ids(topo_idx, operator)
        upstream_input_queues = {upstream_unique_full_op_id: input_queues[upstream_unique_full_op_id] for upstream_unique_full_op_id in upstream_unique_full_op_ids}
        upstream_future_queues = {upstream_unique_full_op_id: future_queues[upstream_unique_full_op_id] for upstream_unique_full_op_id in upstream_unique_full_op_ids}
        return not (self._any_queue_not_empty(upstream_input_queues) or self._any_queue_not_empty(upstream_future_queues))

    def _process_future_results(self, unique_full_op_id: str, future_queues: dict[str, list], plan_stats: PlanStats) -> list[DataRecord]:
        """
        Helper function which takes a full operator id, the future queues, and plan stats, and performs
        the updates to plan stats and progress manager before returning the results from the finished futures.
        """
        # this function is called when the future queue is not empty
        # and the executor is not busy processing other futures
        done_futures, not_done_futures = wait(future_queues[unique_full_op_id], timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

        # add the unfinished futures back to the previous op's future queue
        future_queues[unique_full_op_id] = list(not_done_futures)

        # add the finished futures to the input queue for this operator
        output_records = []
        for future in done_futures:
            record_set: DataRecordSet = future.result()
            records = record_set.data_records
            record_op_stats = record_set.record_op_stats
            num_outputs = sum(record.passed_operator for record in records)

            # update the progress manager
            self.progress_manager.incr(unique_full_op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # update plan stats
            plan_stats.add_record_op_stats(record_op_stats)

            # add records which aren't filtered to the output records
            output_records.extend([record for record in records if record.passed_operator])
        
        return output_records

    def _execute_plan(
            self,
            plan: PhysicalPlan,
            input_queues: dict[str, dict[str, list]],
            future_queues: dict[str, list],
            plan_stats: PlanStats,
        ) -> tuple[list[DataRecord], PlanStats]:
        # process all of the input records using a thread pool
        output_records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            logger.debug(f"Created thread pool with {self.max_workers} workers")

            # execute the plan until either:
            # 1. all records have been processed, or
            # 2. the final limit operation has completed (we break out of the loop if this happens)
            final_op = plan.operator
            while self._any_queue_not_empty(input_queues) or self._any_queue_not_empty(future_queues):
                for topo_idx, operator in enumerate(plan):
                    source_unique_full_op_ids = (
                        [f"source_{operator.get_full_op_id()}"]
                        if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                        else plan.get_source_unique_full_op_ids(topo_idx, operator)
                    )
                    unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"

                    # get any finished futures from the previous operator and add them to the input queue for this operator
                    if not isinstance(operator, (ContextScanOp, ScanPhysicalOp)):
                        for source_unique_full_op_id in source_unique_full_op_ids:
                            records = self._process_future_results(source_unique_full_op_id, future_queues, plan_stats)
                            input_queues[unique_full_op_id][source_unique_full_op_id].extend(records)

                    # for the final operator, add any finished futures to the output_records
                    if unique_full_op_id == f"{topo_idx}-{final_op.get_full_op_id()}":
                        records = self._process_future_results(unique_full_op_id, future_queues, plan_stats)
                        output_records.extend(records)

                    # if this operator does not have enough inputs to execute, then skip it
                    num_inputs = sum(len(inputs) for inputs in input_queues[unique_full_op_id].values())
                    agg_op_not_ready = isinstance(operator, AggregateOp) and not self._upstream_ops_finished(plan, topo_idx, operator, input_queues, future_queues)
                    join_op_not_ready = isinstance(operator, JoinOp) and not self._upstream_ops_finished(plan, topo_idx, operator, input_queues, future_queues)
                    if num_inputs == 0 or agg_op_not_ready or join_op_not_ready:
                        continue

                    # if this operator is an aggregate, process all the records in the input queue
                    if isinstance(operator, AggregateOp):
                        source_unique_full_op_id = source_unique_full_op_ids[0]
                        input_records = [input_queues[unique_full_op_id][source_unique_full_op_id].pop(0) for _ in range(num_inputs)]
                        future = executor.submit(operator, input_records)
                        future_queues[unique_full_op_id].append(future)

                    # if this operator is a join, process all pairs of records from the two input queues
                    elif isinstance(operator, JoinOp):
                        left_unique_full_source_op_id = source_unique_full_op_ids[0]
                        left_num_inputs = len(input_queues[unique_full_op_id][left_unique_full_source_op_id])
                        left_input_records = [input_queues[unique_full_op_id][left_unique_full_source_op_id].pop(0) for _ in range(left_num_inputs)]

                        right_unique_full_source_op_id = source_unique_full_op_ids[1]
                        right_num_inputs = len(input_queues[unique_full_op_id][right_unique_full_source_op_id])
                        right_input_records = [input_queues[unique_full_op_id][right_unique_full_source_op_id].pop(0) for _ in range(right_num_inputs)]

                        future = executor.submit(operator, left_input_records, right_input_records)
                        future_queues[unique_full_op_id].append(future)

                    else:
                        source_unique_full_op_id = source_unique_full_op_ids[0]
                        input_record = input_queues[unique_full_op_id][source_unique_full_op_id].pop(0)
                        future = executor.submit(operator, input_record)
                        future_queues[unique_full_op_id].append(future)

                # break out of loop if the final operator is a LimitScanOp and we've reached its limit
                if isinstance(final_op, LimitScanOp) and len(output_records) == final_op.limit:
                    break

        # finalize plan stats
        plan_stats.finish()

        return output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues and future queues for each operation
        input_queues = self._create_input_queues(plan)
        future_queues = {f"{topo_idx}-{op.get_full_op_id()}": [] for topo_idx, op in enumerate(plan)}

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, num_samples=self.num_samples, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, future_queues, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats


class SequentialParallelExecutionStrategy(ExecutionStrategy):
    """
    A parallel execution strategy that processes operators sequentially.
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

    def _any_queue_not_empty(self, queues: dict[str, list] | dict[str, dict[str, list]]) -> bool:
        """Helper function to check if any queue is not empty."""
        for _, value in queues.items():
            if isinstance(value, dict):
                if any(len(subqueue) > 0 for subqueue in value.values()):
                    return True
            elif len(value) > 0:
                return True
        return False

    def _process_future_results(self, unique_full_op_id: str, future_queues: dict[str, list], plan_stats: PlanStats) -> list[DataRecord]:
        """
        Helper function which takes a full operator id, the future queues, and plan stats, and performs
        the updates to plan stats and progress manager before returning the results from the finished futures.
        """
        # this function is called when the future queue is not empty
        # and the executor is not busy processing other futures
        done_futures, not_done_futures = wait(future_queues[unique_full_op_id], timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

        # add the unfinished futures back to the previous op's future queue
        future_queues[unique_full_op_id] = list(not_done_futures)

        # add the finished futures to the input queue for this operator
        output_records = []
        for future in done_futures:
            record_set: DataRecordSet = future.result()
            records = record_set.data_records
            record_op_stats = record_set.record_op_stats
            num_outputs = sum(record.passed_operator for record in records)

            # update the progress manager
            self.progress_manager.incr(unique_full_op_id, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # update plan stats
            plan_stats.add_record_op_stats(record_op_stats)

            # add records which aren't filtered to the output records
            output_records.extend([record for record in records if record.passed_operator])
        
        return output_records

    def _execute_plan(
            self,
            plan: PhysicalPlan,
            input_queues: dict[str, dict[str, list]],
            future_queues: dict[str, list],
            plan_stats: PlanStats,
        ) -> tuple[list[DataRecord], PlanStats]:
        # process all of the input records using a thread pool
        output_records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            logger.debug(f"Created thread pool with {self.max_workers} workers")

            # execute the plan until either:
            # 1. all records have been processed, or
            # 2. the final limit operation has completed (we break out of the loop if this happens)
            final_op = plan.operator
            for topo_idx, operator in enumerate(plan):
                unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
                source_unique_full_op_ids = plan.get_source_unique_full_op_ids(topo_idx, operator)
                input_queue = input_queues[unique_full_op_id]

                # if this operator is an aggregate, process all the records in the input queue
                if isinstance(operator, AggregateOp):
                    source_unique_full_op_id = source_unique_full_op_ids[0]
                    num_inputs = len(input_queue[unique_full_op_id][source_unique_full_op_id])
                    input_records = [input_queues[unique_full_op_id][source_unique_full_op_id].pop(0) for _ in range(num_inputs)]
                    future = executor.submit(operator, input_records)
                    future_queues[unique_full_op_id].append(future)

                # if this operator is a join, process all pairs of records from the two input queues
                elif isinstance(operator, JoinOp):
                    left_full_source_op_id = source_unique_full_op_ids[0]
                    left_num_inputs = len(input_queues[unique_full_op_id][left_full_source_op_id])
                    left_input_records = [input_queues[unique_full_op_id][left_full_source_op_id].pop(0) for _ in range(left_num_inputs)]

                    right_full_source_op_id = source_unique_full_op_ids[1]
                    right_num_inputs = len(input_queues[unique_full_op_id][right_full_source_op_id])
                    right_input_records = [input_queues[unique_full_op_id][right_full_source_op_id].pop(0) for _ in range(right_num_inputs)]

                    future = executor.submit(operator, left_input_records, right_input_records)
                    future_queues[unique_full_op_id].append(future)

                else:
                    while len(input_queue) > 0:
                        input_record = input_queue.pop(0)
                        future = executor.submit(operator, input_record)
                        future_queues[unique_full_op_id].append(future)

                # block until all futures for this operator have completed; and add finished futures to next operator's input
                while len(future_queues[unique_full_op_id]) > 0:
                    records = self._process_future_results(unique_full_op_id, future_queues, plan_stats)

                    # get any finished futures from the previous operator and add them to the input queue for this operator
                    next_unique_full_op_id = plan.get_next_unique_full_op_id(topo_idx, operator)
                    if next_unique_full_op_id is not None:
                        input_queues[next_unique_full_op_id][unique_full_op_id].extend(records)

                    # for the final operator, add any finished futures to the output_records
                    else:
                        output_records.extend(records)

                        # break out of loop if the final operator is a LimitScanOp and we've reached its limit
                        if isinstance(final_op, LimitScanOp) and len(output_records) == final_op.limit:
                            break

        # finalize plan stats
        plan_stats.finish()

        return output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues and future queues for each operation
        input_queues = self._create_input_queues(plan)
        future_queues = {f"{topo_idx}-{op.get_full_op_id()}": [] for topo_idx, op in enumerate(plan)}

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, num_samples=self.num_samples, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, future_queues, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
