import logging
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, wait

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import OperatorStats, PlanStats
from palimpzest.query.execution.execution_strategy import ExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan

logger = logging.getLogger(__name__)

class PipelinedParallelExecutionStrategy(ExecutionStrategy):
    """
    A parallel execution strategy that processes data through a pipeline of operators using thread-based parallelism.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = (
            self.get_parallel_max_workers()
            if self.max_workers is None
            else self.max_workers
        )

    def get_parallel_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return max(int(0.8 * multiprocessing.cpu_count()), 1)

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

        # initialize data structures to help w/processing DAG
        processing_queue = []
        op_id_to_futures_in_flight = {op.get_op_id(): 0 for op in plan.operators}
        op_id_to_operator = {op.get_op_id(): op for op in plan.operators}
        op_id_to_prev_operator = {
            op.get_op_id(): plan.operators[idx - 1] if idx > 0 else None for idx, op in enumerate(plan.operators)
        }
        op_id_to_next_operator = {
            op.get_op_id(): plan.operators[idx + 1] if idx + 1 < len(plan.operators) else None
            for idx, op in enumerate(plan.operators)
        }
        op_id_to_op_idx = {op.get_op_id(): idx for idx, op in enumerate(plan.operators)}

        # get handle to scan operator and pre-compute its op_id and size
        source_operator = plan.operators[0]
        assert isinstance(source_operator, ScanPhysicalOp), "First operator in physical plan must be a ScanPhysicalOp"
        source_op_id = source_operator.get_op_id()
        datareader_len = len(source_operator.datareader)

        # get limit of final limit operator (if one exists)
        final_limit = plan.operators[-1].limit if isinstance(plan.operators[-1], LimitScanOp) else None

        # create thread pool w/max workers
        futures = []
        current_scan_idx = self.scan_start_idx
        with ThreadPoolExecutor(max_workers=plan_workers) as executor:
            logger.debug(f"Created thread pool with {plan_workers} workers")
            # create initial (set of) future(s) to read first source record;
            futures.append(executor.submit(PhysicalOperator.execute_op_wrapper, source_operator, current_scan_idx))
            op_id_to_futures_in_flight[source_op_id] += 1
            current_scan_idx += 1

            # iterate until we have processed all operators on all records or come to an early stopping condition
            while len(futures) > 0:
                # get the set of futures that have (and have not) finished in the last PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

                # cast not_done_futures from a set to a list so we can append to it
                not_done_futures = list(not_done_futures)

                # process finished futures, creating new ones as needed
                new_futures = []
                for future in done_futures:
                    # get the result
                    record_set, operator, _ = future.result()
                    op_id = operator.get_op_id()
                    logger.debug(f"Processed future for operator {op_id} with {len(record_set)} records")

                    # decrement future from mapping of futures in-flight
                    op_id_to_futures_in_flight[op_id] -= 1

                    # update plan stats
                    prev_operator = op_id_to_prev_operator[op_id]
                    plan_stats.operator_stats[op_id].add_record_op_stats(
                        record_set.record_op_stats,
                        source_op_id=prev_operator.get_op_id() if prev_operator is not None else None,
                        plan_id=plan.plan_id,
                    )

                    # process each record output by the future's operator
                    for record in record_set:
                        # skip records which are filtered out
                        if not getattr(record, "passed_operator", True):
                            continue

                        # add records (which are not filtered) to the cache, if allowed
                        if self.cache:
                            # self.datadir.append_cache(operator.target_cache_id, record)
                            pass

                        # add records to processing queue if there is a next_operator; otherwise add to output_records
                        next_operator = op_id_to_next_operator[op_id]
                        if next_operator is not None:
                            processing_queue.append((next_operator, record))
                        else:
                            output_records.append(record)

                    # if this operator was a source scan, update the number of source records scanned
                    if op_id == source_op_id:
                        source_records_scanned += len(record_set)

                        # scan next record if we can still draw records from source
                        if source_records_scanned < num_samples and current_scan_idx < datareader_len:
                            new_futures.append(executor.submit(PhysicalOperator.execute_op_wrapper, source_operator, current_scan_idx))
                            op_id_to_futures_in_flight[source_op_id] += 1
                            current_scan_idx += 1

                    # check early stopping condition based on final limit
                    if final_limit is not None and len(output_records) >= final_limit:
                        output_records = output_records[:final_limit]
                        futures = []
                        break

                # process all records in the processing queue which are ready to be executed
                temp_processing_queue = []
                for operator, candidate in processing_queue:
                    # if the candidate is not an input to an aggregate, execute it right away
                    if not isinstance(operator, AggregateOp):
                        future = executor.submit(PhysicalOperator.execute_op_wrapper, operator, candidate)
                        new_futures.append(future)
                        op_id_to_futures_in_flight[operator.get_op_id()] += 1

                    # otherwise, put it back on the queue
                    else:
                        temp_processing_queue.append((operator, candidate))

                # any remaining candidates are inputs to aggregate operators; for each aggregate operator
                # determine if it is ready to execute -- and execute all of its candidates if so
                processing_queue = []
                agg_op_ids = set([operator.get_op_id() for operator, _ in temp_processing_queue])
                for agg_op_id in agg_op_ids:
                    agg_op_idx = op_id_to_op_idx[agg_op_id]

                    # compute if all upstream operators' processing queues are empty and their in-flight futures are finished
                    upstream_ops_are_finished = True
                    for upstream_op_idx in range(agg_op_idx):
                        upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                        upstream_op_id_queue = list(
                            filter(lambda tup: tup[0].get_op_id() == upstream_op_id, temp_processing_queue)
                        )

                        upstream_ops_are_finished = (
                            upstream_ops_are_finished
                            and len(upstream_op_id_queue) == 0
                            and op_id_to_futures_in_flight[upstream_op_id] == 0
                        )

                    # get the subset of candidates for this aggregate operator
                    candidate_tuples = list(filter(lambda tup: tup[0].get_op_id() == agg_op_id, temp_processing_queue))

                    # execute the operator on the candidates if it's ready
                    if upstream_ops_are_finished:
                        operator = op_id_to_operator[agg_op_id]
                        candidates = list(map(lambda tup: tup[1], candidate_tuples))
                        future = executor.submit(PhysicalOperator.execute_op_wrapper, operator, candidates)
                        new_futures.append(future)
                        op_id_to_futures_in_flight[operator.get_op_id()] += 1

                    # otherwise, add the candidates back to the processing queue
                    else:
                        processing_queue.extend(candidate_tuples)

                # update list of futures
                not_done_futures.extend(new_futures)
                futures = not_done_futures

        # if caching was allowed, close the cache
        if self.cache:
            for _ in plan.operators:
                # self.datadir.close_cache(operator.target_cache_id)
                pass

        logger.info(f"Completed execution of plan {plan.plan_id} in {time.time() - plan_start_time:.2f} seconds")
        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)
        logger.info(f"Completed execution of plan {plan.plan_id} in {time.time() - plan_start_time:.2f} seconds")
        logger.debug(f"Plan execution stats: (plan_str={plan_stats.plan_str}, plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return output_records, plan_stats
