import logging

from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
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

    def _execute_plan(self, plan: PhysicalPlan, input_queues: dict[str, dict[str, list]], plan_stats: PlanStats) -> tuple[list[DataRecord], PlanStats]:
        # execute the plan one operator at a time
        output_records = []
        for topo_idx, operator in enumerate(plan):
            # if we've filtered out all records, terminate early
            source_unique_full_op_ids = (
                [f"source_{operator.get_full_op_id()}"]
                if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                else plan.get_source_unique_full_op_ids(topo_idx, operator)
            )
            unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"
            num_inputs = sum(len(input_queues[unique_full_op_id][source_unique_full_op_id]) for source_unique_full_op_id in source_unique_full_op_ids)
            if num_inputs == 0:
                break

            # begin to process this operator
            records, record_op_stats = [], []
            logger.info(f"Processing operator {operator.op_name()} ({unique_full_op_id})")

            # if this operator is an aggregate, process all the records in the input_queue
            if isinstance(operator, AggregateOp):
                source_unique_full_op_id = source_unique_full_op_ids[0]
                record_set = operator(candidates=input_queues[unique_full_op_id][source_unique_full_op_id])
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats
                num_outputs = sum(record._passed_operator for record in records)

                # update the progress manager
                self.progress_manager.incr(unique_full_op_id, num_inputs=1, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # if this operator is a join, process all pairs of records from the two input queues
            elif isinstance(operator, JoinOp):
                left_full_source_op_id = source_unique_full_op_ids[0]
                left_num_inputs = len(input_queues[unique_full_op_id][left_full_source_op_id])
                left_input_records = [input_queues[unique_full_op_id][left_full_source_op_id].pop(0) for _ in range(left_num_inputs)]

                right_full_source_op_id = source_unique_full_op_ids[1]
                right_num_inputs = len(input_queues[unique_full_op_id][right_full_source_op_id])
                right_input_records = [input_queues[unique_full_op_id][right_full_source_op_id].pop(0) for _ in range(right_num_inputs)]

                record_set, num_inputs_processed = operator(left_input_records, right_input_records)
                records = record_set.data_records
                record_op_stats = record_set.record_op_stats

                # process the join one last time with final=True to handle any left/right/outer join logic
                if operator.how in ("left", "right", "outer"):
                    record_set, num_inputs_processed = operator([], [], final=True)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)
      
                num_outputs = sum(record._passed_operator for record in records)

                # update the progress manager
                self.progress_manager.incr(unique_full_op_id, num_inputs=num_inputs_processed, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

            # otherwise, process the records in the input queue for this operator one at a time
            else:
                source_unique_full_op_id = source_unique_full_op_ids[0]
                for input_record in input_queues[unique_full_op_id][source_unique_full_op_id]:
                    record_set = operator(input_record)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)
                    num_outputs = sum(record._passed_operator for record in record_set.data_records)

                    # update the progress manager
                    self.progress_manager.incr(unique_full_op_id, num_inputs=1, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                    # finish early if this is a limit
                    if isinstance(operator, LimitScanOp) and len(records) == operator.limit:
                        break

            # update plan stats
            plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

            # update next input_queue (if it exists)
            output_records = [record for record in records if record._passed_operator]
            next_unique_full_op_id = plan.get_next_unique_full_op_id(topo_idx, operator)
            if next_unique_full_op_id is not None:
                input_queues[next_unique_full_op_id][unique_full_op_id] = output_records

            logger.info(f"Finished processing operator {operator.op_name()} ({unique_full_op_id}), and generated {len(records)} records")

        # finalize plan stats
        plan_stats.finish()

        return output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues for each operation
        input_queues = self._create_input_queues(plan)

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, num_samples=self.num_samples, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, plan_stats)

        finally:
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

    def _any_queue_not_empty(self, queues: dict[str, list] | dict[str, dict[str, list]]) -> bool:
        """Helper function to check if any queue is not empty."""
        for _, value in queues.items():
            if isinstance(value, dict):
                if any(len(subqueue) > 0 for subqueue in value.values()):
                    return True
            elif len(value) > 0:
                return True
        return False

    def _upstream_ops_finished(self, plan: PhysicalPlan, unique_full_op_id: str, input_queues: dict[str, dict[str, list]]) -> bool:
        """Helper function to check if agg / join operator is ready to process its inputs."""
        upstream_unique_full_op_ids = plan.get_upstream_unique_full_op_ids(unique_full_op_id)
        upstream_input_queues = {upstream_unique_full_op_id: input_queues[upstream_unique_full_op_id] for upstream_unique_full_op_id in upstream_unique_full_op_ids}
        return not self._any_queue_not_empty(upstream_input_queues)


    def _execute_plan(self, plan: PhysicalPlan, input_queues: dict[str, dict[str, list]], plan_stats: PlanStats) -> tuple[list[DataRecord], PlanStats]:
        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed (we break out of the loop if this happens)
        final_output_records = []
        while self._any_queue_not_empty(input_queues):
            for topo_idx, operator in enumerate(plan):
                # if this operator does not have enough inputs to execute, then skip it
                source_unique_full_op_ids = (
                    [f"source_{operator.get_full_op_id()}"]
                    if isinstance(operator, (ContextScanOp, ScanPhysicalOp))
                    else plan.get_source_unique_full_op_ids(topo_idx, operator)
                )
                unique_full_op_id = f"{topo_idx}-{operator.get_full_op_id()}"

                num_inputs = sum(len(input_queues[unique_full_op_id][source_unique_full_op_id]) for source_unique_full_op_id in source_unique_full_op_ids)
                agg_op_not_ready = isinstance(operator, AggregateOp) and not self._upstream_ops_finished(plan, unique_full_op_id, input_queues)
                join_op_not_ready = isinstance(operator, JoinOp) and not self._upstream_ops_finished(plan, unique_full_op_id, input_queues)
                if num_inputs == 0 or agg_op_not_ready or join_op_not_ready:
                    continue

                # create empty lists for records and execution stats generated by executing this operator on its next input(s)
                records, record_op_stats = [], []

                # if the next operator is an aggregate, process all the records in the input_queue
                if isinstance(operator, AggregateOp):
                    source_unique_full_op_id = source_unique_full_op_ids[0]
                    input_records = [input_queues[unique_full_op_id][source_unique_full_op_id].pop(0) for _ in range(num_inputs)]
                    record_set = operator(candidates=input_records)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    num_outputs = sum(record._passed_operator for record in records)

                    # update the progress manager
                    self.progress_manager.incr(unique_full_op_id, num_inputs=1, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                # if this operator is a join, process all pairs of records from the two input queues
                elif isinstance(operator, JoinOp):
                    left_full_source_op_id = source_unique_full_op_ids[0]
                    left_num_inputs = len(input_queues[unique_full_op_id][left_full_source_op_id])
                    left_input_records = [input_queues[unique_full_op_id][left_full_source_op_id].pop(0) for _ in range(left_num_inputs)]

                    right_full_source_op_id = source_unique_full_op_ids[1]
                    right_num_inputs = len(input_queues[unique_full_op_id][right_full_source_op_id])
                    right_input_records = [input_queues[unique_full_op_id][right_full_source_op_id].pop(0) for _ in range(right_num_inputs)]

                    record_set, num_inputs_processed = operator(left_input_records, right_input_records)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    num_outputs = sum(record._passed_operator for record in records)

                    # update the progress manager
                    self.progress_manager.incr(unique_full_op_id, num_inputs=num_inputs_processed, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                # otherwise, process the next record in the input queue for this operator
                else:
                    source_unique_full_op_id = source_unique_full_op_ids[0]
                    input_record = input_queues[unique_full_op_id][source_unique_full_op_id].pop(0)
                    record_set = operator(input_record)
                    records = record_set.data_records
                    record_op_stats = record_set.record_op_stats
                    num_outputs = sum(record._passed_operator for record in records)

                    # update the progress manager
                    self.progress_manager.incr(unique_full_op_id, num_inputs=1, num_outputs=num_outputs, total_cost=record_set.get_total_cost())

                # if this is a join operator with no more inputs to process, then finish it
                if isinstance(operator, JoinOp) and operator.how in ("left", "right", "outer"):
                    join_op_upstream_finished = self._upstream_ops_finished(plan, unique_full_op_id, input_queues)
                    join_input_queues_empty = all(len(inputs) == 0 for inputs in input_queues[unique_full_op_id].values())
                    if join_op_upstream_finished and join_input_queues_empty and not operator.finished:
                        # process the join one last time with final=True to handle any left/right/outer join logic
                        record_set, num_inputs_processed = operator([], [], final=True)
                        records.extend(record_set.data_records)
                        record_op_stats.extend(record_set.record_op_stats)
                        num_outputs += sum(record._passed_operator for record in record_set.data_records)
                        operator.set_finished()

                # update plan stats
                plan_stats.add_record_op_stats(unique_full_op_id, record_op_stats)

                # update next input_queue or final_output_records
                output_records = [record for record in records if record._passed_operator]
                next_unique_full_op_id = plan.get_next_unique_full_op_id(topo_idx, operator)
                if next_unique_full_op_id is not None:
                    input_queues[next_unique_full_op_id][unique_full_op_id].extend(output_records)
                else:
                    final_output_records.extend(output_records)

                logger.info(f"Finished processing operator {operator.op_name()} ({unique_full_op_id}) on {num_inputs} records")

            # break out of loop if the final operator is a LimitScanOp and we've reached its limit
            if isinstance(plan.operator, LimitScanOp) and len(final_output_records) == plan.operator.limit:
                break

        # finalize plan stats
        plan_stats.finish()

        return final_output_records, plan_stats

    def execute_plan(self, plan: PhysicalPlan):
        """Initialize the stats and execute the plan."""
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = PlanStats.from_plan(plan)
        plan_stats.start()

        # initialize input queues for each operation
        input_queues = self._create_input_queues(plan)        

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, self.num_samples, self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail
        #       because the progress manager cannot get a handle to the console 
        try:
            # execute plan
            output_records, plan_stats = self._execute_plan(plan, input_queues, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
