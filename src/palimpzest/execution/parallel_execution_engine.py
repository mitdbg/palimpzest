from palimpzest.constants import PlanType, PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import DataRecord
from palimpzest.execution import ExecutionEngine
from palimpzest.operators import AggregateOp, LimitScanOp, MarshalAndScanDataOp, PhysicalOperator
from palimpzest.planner import PhysicalPlan

from palimpzest.dataclasses import OperatorStats, PlanStats

from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional, Union

import time


class PipelinedParallelExecutionEngine(ExecutionEngine):
    """
    This class implements the abstract execute_plan() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute() method.
    """

    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: Union[DataRecord, List[DataRecord]]):
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        records, record_op_stats_lst = operator(op_input)

        return records, record_op_stats_lst, operator

    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"{plan_type.value} {str(plan_idx)}:")
            plan.printPlan()
            print("---")

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.get_plan_id())
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            plan_stats.operator_stats[op_id] = OperatorStats(op_idx=op_idx, op_id=op_id, op_name=op.op_name()) # TODO: also add op_details here

        # set limit on the number of samples if this is a sentinel plan
        num_samples = self.num_samples if plan_type == PlanType.SENTINEL else float("inf")  

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0

        # initialize data structures to help w/processing DAG
        processing_queue = []
        op_id_to_futures_in_flight = {op.get_op_id(): 0 for op in plan.operators}
        op_id_to_prev_operator = {
            op.get_op_id(): plan.operators[idx - 1] if idx > 0 else None
            for idx, op in enumerate(plan.operators)
        }
        op_id_to_next_operator = {
            op.get_op_id(): plan.operators[idx + 1] if idx + 1 < len(plan.operators) else None
            for idx, op in enumerate(plan.operators)
        }

        # get handle to DataSource and pre-compute its op_id and size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(self.source_dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.cachedDataIdentifier)
        )
        source_op_id = source_operator.get_op_id()
        datasource_len = len(datasource)

        # compute op_id and limit of final limit operator (if one exists)
        final_limit = plan.operators[-1].limit if isinstance(plan.operators[-1], LimitScanOp) else None

        # create thread pool w/max workers
        futures = []
        current_scan_idx = self.scan_start_idx
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # create initial set of futures to read first source file;
            # construct input DataRecord for DataSourcePhysicalOp
            candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
            candidate.idx = current_scan_idx
            candidate.get_item_fn = datasource.getItem
            candidate.cardinality = datasource.cardinality
            futures.append(executor.submit(PipelinedParallelExecutionEngine.execute_op_wrapper, source_operator, candidate))
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
                    records, record_op_stats_lst, operator = future.result()
                    op_id = operator.get_op_id()

                    # decrement future from mapping of futures in-flight
                    op_id_to_futures_in_flight[op_id] -= 1

                    # update plan stats
                    op_stats = plan_stats.operator_stats[op_id]
                    for record_op_stats in record_op_stats_lst:
                        # TODO code a nice __add__ function for OperatorStats and RecordOpStats
                        prev_operator = op_id_to_prev_operator[op_id]
                        record_op_stats.source_op_id = prev_operator.get_op_id() if prev_operator is not None else None
                        record_op_stats.plan_id = plan.get_plan_id()
                        op_stats.record_op_stats_lst.append(record_op_stats)
                        op_stats.total_op_time += record_op_stats.time_per_record
                        op_stats.total_op_cost += record_op_stats.cost_per_record

                    plan_stats.operator_stats[op_id] = op_stats

                    # process each record output by the future's operator
                    for record in records:
                        # skip records which are filtered out
                        if not getattr(record, "_passed_filter", True):
                            continue

                        # add records (which are not filtered) to the cache, if allowed
                        if not self.nocache:
                            self.datadir.appendCache(operator.targetCacheId, record)

                        # add records to processing queue if there is a next_operator; otherwise add to output_records
                        next_operator = op_id_to_next_operator[op_id]
                        if next_operator is not None:
                            processing_queue.append((next_operator, record))
                        else:
                            output_records.append(record)

                    # if this operator was a source scan, update the number of source records scanned
                    if op_id == source_op_id:
                        source_records_scanned += len(records)

                        # scan next record if we can still draw records from source
                        if source_records_scanned < num_samples and current_scan_idx < datasource_len:
                            # construct input DataRecord for DataSourcePhysicalOp
                            candidate = DataRecord(schema=SourceRecord, parent_uuid=None, scan_idx=current_scan_idx)
                            candidate.idx = current_scan_idx
                            candidate.get_item_fn = datasource.getItem
                            candidate.cardinality = datasource.cardinality
                            new_futures.append(executor.submit(PipelinedParallelExecutionEngine.execute_op_wrapper, source_operator, candidate))
                            op_id_to_futures_in_flight[source_op_id] += 1
                            current_scan_idx += 1

                    # check early stopping condition based on final limit
                    if final_limit is not None and len(output_records) >= final_limit:
                        output_records = output_records[:final_limit]
                        futures = []
                        break

                    # only invoke aggregate operator(s) once all upstream operators' processing queues are empty
                    # and their in-flight futures are finished
                    if isinstance(operator, AggregateOp):
                        this_op_idx = 0
                        while op_id != plan.operators[this_op_idx].get_op_id():
                            this_op_idx += 1

                        upstream_ops_are_finished = True
                        for upstream_op_idx in range(this_op_idx):
                            upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                            upstream_op_id_queue = list(filter(lambda tup: tup[0].get_op_id() == upstream_op_id, processing_queue))

                            upstream_ops_are_finished = (
                                upstream_ops_are_finished
                                and len(upstream_op_id_queue) == 0
                                and op_id_to_futures_in_flight[upstream_op_id] == 0
                            )

                        if upstream_ops_are_finished:
                            candidates = list(filter(lambda tup: tup[0].get_op_id() == op_id, processing_queue))
                            candidates = list(map(lambda tup: tup[1], candidates))
                            future = executor.submit(PipelinedParallelExecutionEngine.execute_op_wrapper, operator, candidates)
                            new_futures.append(future)
                            op_id_to_futures_in_flight[op_id] += 1
                            processing_queue = list(filter(lambda tup: tup[0].get_op_id() != op_id, processing_queue))

                    # otherwise, process all the records in the processing queue
                    else:
                        for operator, candidate in processing_queue:
                            future = executor.submit(PipelinedParallelExecutionEngine.execute_op_wrapper, operator, candidate)
                            new_futures.append(future)
                            op_id_to_futures_in_flight[op_id] += 1

                        processing_queue = []

                # update list of futures
                not_done_futures.extend(new_futures)
                futures = not_done_futures

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats
