import logging
from concurrent.futures import ThreadPoolExecutor, wait

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.distinct import DistinctOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.limit import LimitScanOp
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

    def _any_queue_not_empty(self, queues: dict[str, list] | dict[str, dict[str, list]]) -> bool:
        """Helper function to check if any queue is not empty."""
        for _, value in queues.items():
            if isinstance(value, dict):
                if any(len(subqueue) > 0 for subqueue in value.values()):
                    return True
            elif len(value) > 0:
                return True
        return False

    def _upstream_ops_finished(self, plan: PhysicalPlan, unique_full_op_id: str, input_queues: dict[str, dict[str, list]], future_queues: dict[str, list]) -> bool:
        """Helper function to check if agg / join operator is ready to process its inputs."""
        upstream_unique_full_op_ids = plan.get_upstream_unique_full_op_ids(unique_full_op_id)
        upstream_input_queues = {upstream_unique_full_op_id: input_queues[upstream_unique_full_op_id] for upstream_unique_full_op_id in upstream_unique_full_op_ids}
        upstream_future_queues = {upstream_unique_full_op_id: future_queues[upstream_unique_full_op_id] for upstream_unique_full_op_id in upstream_unique_full_op_ids}
        return not (self._any_queue_not_empty(upstream_input_queues) or self._any_queue_not_empty(upstream_future_queues))

    def _finish_outer_join(self, executor: ThreadPoolExecutor, plan: PhysicalPlan, unique_full_op_id: str, input_queues: dict[str, dict[str, list]], future_queues: dict[str, list]) -> None:
        join_op_upstream_finished = self._upstream_ops_finished(plan, unique_full_op_id, input_queues, future_queues)
        join_input_queues_empty = all(len(inputs) == 0 for inputs in input_queues[unique_full_op_id].values())
        join_future_queue_empty = len(future_queues[unique_full_op_id]) == 0
        if join_op_upstream_finished and join_input_queues_empty and join_future_queue_empty:
            # process the join one last time with final=True to handle any left/right/outer join logic
            operator = self.unique_full_op_id_to_operator[unique_full_op_id]
            if not operator.finished:
                def finalize_op(operator):
                    return operator([], [], final=True)
                future = executor.submit(finalize_op, operator)
                future_queues[unique_full_op_id].append(future)
                operator.set_finished()

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
        output_records, total_inputs_processed, total_cost = [], 0, 0.0
        for future in done_futures:
            output = future.result()
            record_set, num_inputs_processed = output if self.is_join_op[unique_full_op_id] else (output, 1)

            # record set can be empty if one side of join has no input records yet
            if len(record_set) == 0:
                continue

            # otherwise, process records and their stats
            records = record_set.data_records
            record_op_stats = record_set.record_op_stats

            # update the inputs processed and total cost
            total_inputs_processed += num_inputs_processed
            total_cost += record_set.get_total_cost()

            # update plan stats
            plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

            # add records which aren't filtered to the output records
            output_records.extend([record for record in records if record._passed_operator])

        # update the progress manager
        if total_inputs_processed > 0:
            num_outputs = len(output_records)
            self.progress_manager.incr(unique_full_op_id, num_inputs=total_inputs_processed, num_outputs=num_outputs, total_cost=total_cost)

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

                            # if the source is a left/right/outer join operator with no more inputs to process, then finish it
                            if self.is_outer_join_op[source_unique_full_op_id]:
                                self._finish_outer_join(executor, plan, source_unique_full_op_id, input_queues, future_queues)

                    # for the final operator, add any finished futures to the output_records
                    if unique_full_op_id == f"{topo_idx}-{final_op.get_full_op_id()}":
                        records = self._process_future_results(unique_full_op_id, future_queues, plan_stats)
                        output_records.extend(records)

                        # if this is a left/right/outer join operator with no more inputs to process, then finish it
                        if self.is_outer_join_op[unique_full_op_id]:
                            self._finish_outer_join(executor, plan, unique_full_op_id, input_queues, future_queues)

                    # if this operator does not have enough inputs to execute, then skip it
                    num_inputs = sum(len(inputs) for inputs in input_queues[unique_full_op_id].values())
                    agg_op_not_ready = isinstance(operator, AggregateOp) and not self._upstream_ops_finished(plan, unique_full_op_id, input_queues, future_queues)
                    join_op_not_ready = isinstance(operator, JoinOp) and not self.join_has_downstream_limit_op[unique_full_op_id] and not self._upstream_ops_finished(plan, unique_full_op_id, input_queues, future_queues)
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

                        # NOTE: it would be nice to use executor for join inputs here; but for now synchronizing may be necessary
                        # future = executor.submit(operator, left_input_records, right_input_records)
                        # future_queues[unique_full_op_id].append(future)
                        record_set, num_inputs_processed = operator(left_input_records, right_input_records)
                        def no_op(rset, num_inputs_processed):
                            return rset, num_inputs_processed
                        future = executor.submit(no_op, record_set, num_inputs_processed)
                        future_queues[unique_full_op_id].append(future)

                    # if this operator is a limit, process one record at a time
                    elif isinstance(operator, LimitScanOp):
                        source_unique_full_op_id = source_unique_full_op_ids[0]
                        num_records_to_process = min(len(input_queues[unique_full_op_id][source_unique_full_op_id]), operator.limit - len(output_records))
                        for _ in range(num_records_to_process):
                            input_record = input_queues[unique_full_op_id][source_unique_full_op_id].pop(0)
                            future = executor.submit(operator, input_record)
                            future_queues[unique_full_op_id].append(future)

                        # if this is the final operator, add any finished futures to the output_records
                        # immediately so that we can break out of the loop if we've reached the limit
                        if unique_full_op_id == f"{topo_idx}-{final_op.get_full_op_id()}":
                            records = self._process_future_results(unique_full_op_id, future_queues, plan_stats)
                            output_records.extend(records)

                    # if this operator is a distinct, process records sequentially
                    # (distinct is not parallelized because it requires maintaining a set of seen records)
                    elif isinstance(operator, DistinctOp):
                        source_unique_full_op_id = source_unique_full_op_ids[0]
                        input_records = input_queues[unique_full_op_id][source_unique_full_op_id]
                        for record in input_records:
                            record_set = operator(record)
                            def no_op(rset):
                                return rset
                            future = executor.submit(no_op, record_set)
                            future_queues[unique_full_op_id].append(future)

                        # clear the input queue for this operator since we processed all records
                        input_queues[unique_full_op_id][source_unique_full_op_id].clear()

                    # otherwise, process records according to batch size
                    else:
                        source_unique_full_op_id = source_unique_full_op_ids[0]
                        input_records = input_queues[unique_full_op_id][source_unique_full_op_id]
                        if self.batch_size is None:
                            for input_record in input_records:
                                future = executor.submit(operator, input_record)
                                future_queues[unique_full_op_id].append(future)
                            input_queues[unique_full_op_id][source_unique_full_op_id].clear()
                        else:
                            batch_size = min(self.batch_size, len(input_records))
                            batch_input_records = input_records[:batch_size]
                            for input_record in batch_input_records:
                                future = executor.submit(operator, input_record)
                                future_queues[unique_full_op_id].append(future)
                            input_queues[unique_full_op_id][source_unique_full_op_id] = input_records[batch_size:]

                # TODO: change logic to stop upstream operators once a limit is reached
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

        # precompute which operators are (outer) joins and which joins have downstream limit ops
        self.is_join_op = {f"{topo_idx}-{op.get_full_op_id()}": isinstance(op, JoinOp) for topo_idx, op in enumerate(plan)}
        self.is_outer_join_op = {f"{topo_idx}-{op.get_full_op_id()}": isinstance(op, JoinOp) and op.how in ("left", "right", "outer") for topo_idx, op in enumerate(plan)}
        self.join_has_downstream_limit_op = {}
        for topo_idx, op in enumerate(plan):
            if isinstance(op, JoinOp):
                unique_full_op_id = f"{topo_idx}-{op.get_full_op_id()}"
                has_downstream_limit_op = False
                for inner_topo_idx, op in enumerate(plan):
                    if inner_topo_idx <= topo_idx:
                        continue
                    if isinstance(op, LimitScanOp):
                        has_downstream_limit_op = True
                        break
                self.join_has_downstream_limit_op[unique_full_op_id] = has_downstream_limit_op

        # precompute mapping from unique_full_op_id to operator instance
        self.unique_full_op_id_to_operator = {f"{topo_idx}-{op.get_full_op_id()}": op for topo_idx, op in enumerate(plan)}

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
