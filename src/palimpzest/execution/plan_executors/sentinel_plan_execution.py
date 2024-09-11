from palimpzest.constants import MAX_ID_CHARS, PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS, PickOutputStrategy
from palimpzest.corelib.schemas import Schema, SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats, RecordOpStats
from palimpzest.elements import DataRecord
from palimpzest.execution import ExecutionEngine
from palimpzest.operators import (
    AggregateOp,
    DataSourcePhysicalOp,
    FilterOp,
    LimitScanOp,
    LLMConvertBonded,
    LLMFilter,
    MarshalAndScanDataOp,
    PhysicalOperator,
)
from palimpzest.optimizer import SentinelPlan
from palimpzest.utils import create_sample_matrix, getChampionModel

from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Tuple, Union

import numpy as np

import hashlib
import time


class SequentialSingleThreadSentinelPlanExecutor(ExecutionEngine):
    """
    This class implements the abstract execute_plan() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute() method.
    """
    def __init__(self, pick_output_strategy: PickOutputStrategy = PickOutputStrategy.CHAMPION, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 1 if self.max_workers is None else self.max_workers
        self.pick_output_fn = (
            self.pick_champion_output
            if pick_output_strategy == PickOutputStrategy.CHAMPION
            else self.pick_ensemble_output
        )

    def compute_op_set_id(self, op_set: List[PhysicalOperator]):
        hash_str = str(tuple(op.get_op_id() for op in op_set))
        return hashlib.sha256(hash_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

    def pick_champion_output(self, record_outputs):
        # if there's only one operator in the set, we return its records
        if len(record_outputs) == 1:
            records, _ = record_outputs[0]
            return records

        # get the champion model
        champion_model = getChampionModel()

        # return output from the bonded query or filter with the best model;
        # ignore response from all other operations
        out_records = []
        for records, operator in record_outputs:
            if isinstance(operator, LLMConvertBonded) and operator.model == champion_model:
                out_records = records
                break

            if isinstance(operator, LLMFilter) and operator.model == champion_model:
                out_records = records
                break

        return out_records

    def pick_ensemble_output(self, record_outputs):
        # if there's only one operator in the set, we return its records
        if len(record_outputs) == 1:
            records, _ = record_outputs[0]
            return records

        # aggregate records at each index in the response
        idx_to_records = {}
        for records, _ in record_outputs:
            for idx, record in enumerate(records):
                if idx not in idx_to_records:
                    idx_to_records[idx] = [record]
                else:
                    idx_to_records.append(record)

        # output most common answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records = idx_to_records[idx]
            most_common_record = max(set(records), key=records.count)
            out_records.append(most_common_record)

        return out_records


    def execute_op_set(self, input_records: List[DataRecord], op_set: List[PhysicalOperator]) -> Tuple[List[DataRecord], List[RecordOpStats], np.array]:
        """
        Return List[record], List[source_record_idx], List[record_op_stats]
        """
        # initialize output data structures
        records, record_op_stats_lst = [], []

        # create a sample matrix
        sample_matrix = create_sample_matrix(len(input_records), len(op_set), self.rank)

        # handle aggregate operators
        if isinstance(op_set[0], AggregateOp):
            # NOTE: will need to change this if we ever have competing aggregate implemenations
            operator = op_set[0]
            out_records, out_record_op_stats_lst = operator(input_records)
            records.extend(out_records)
            record_op_stats_lst.extend(out_record_op_stats_lst)
            return records, record_op_stats_lst, sample_matrix

        # handle limit; there is only a single implementation of a limit
        if isinstance(op_set[0], LimitScanOp):
            limit_op = op_set[0]
            for record in input_records:
                out_records, out_record_op_stats_lst = limit_op(record)
                records.extend(out_records)
                record_op_stats_lst.extend(out_record_op_stats_lst)

                if len(records) == limit_op.limit:
                    return records, record_op_stats_lst, sample_matrix

        # run operator set on input records
        for record_idx, record in enumerate(input_records):
            op_set_out_records = []
            for op_idx, operator in enumerate(op_set):
                if sample_matrix[record_idx, op_idx]:
                    # run operator on candidate record
                    out_records, out_record_op_stats_lst = operator(record.copy())

                    # immediately add record op stats to list of all record op stats
                    record_op_stats_lst.extend(out_record_op_stats_lst)

                    # add output record(s), record index, and op info to list of outputs
                    op_set_out_records.append((out_records, operator))

            # select the output records
            op_set_out_records = self.pick_output_fn(op_set_out_records)
            records.extend(op_set_out_records)

        return records, record_op_stats_lst, sample_matrix


    def execute_plan(self, plan: SentinelPlan,
                     scan_start_idx: int,
                     scan_end_idx: int,
                     plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (records=[{scan_start_idx}:{scan_end_idx}]):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op_set in plan.operator_sets:
            op_set_id = self.compute_op_set_id(op_set)
            op_set_str = ",".join([op.op_name() for op in op_set])
            op_set_name = f"OpSet({op_set_str})"
            op_set_details = {
                op.op_name(): {k: v for k, v in op.get_op_params().items() if k not in ["inputSchema", "outputSchema"]}
                for op in op_set
            }
            plan_stats.operator_stats[op_set_id] = OperatorStats(op_id=op_set_id, op_name=op_set_name, op_details=op_set_details)

        # initialize list of output records and intermediate variables
        output_records = []
        current_scan_idx = scan_start_idx

        # get handle to DataSource (# making the assumption that first operator_set can only be a scan
        source_operator = plan.operator_sets[0][0]
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )

        # initialize processing queues for each operation set
        processing_queues = {
            self.compute_op_set_id(op_set): []
            for op_set in plan.operator_sets
            if not isinstance(op_set[0], DataSourcePhysicalOp)
        }

        # execute the plan one operator_set at a time
        for op_set_idx, op_set in enumerate(plan.operator_sets):
            op_set_id = self.compute_op_set_id(op_set)
            prev_op_set_id = (
                self.compute_op_set_id(plan.operator_sets[op_set_idx - 1])
                if op_set_idx > 1
                else None
            )
            next_op_set_id = (
                self.compute_op_set_id(plan.operator_sets[op_set_idx + 1])
                if op_set_idx + 1 < len(plan.operator_sets)
                else None
            )

            # initialize output records and record_op_stats_lst for this operator
            records, record_op_stats_lst, sample_matrix = [], [], None

            # invoke datasource operator(s) until we run out of source records or hit the num_samples limit
            if isinstance(op_set[0], DataSourcePhysicalOp):
                # construct set of input records
                candidates = []
                keep_scanning_source_records = True
                while keep_scanning_source_records:
                    # construct input DataRecord for DataSourcePhysicalOp
                    candidate = DataRecord(schema=SourceRecord, parent_id=None, scan_idx=current_scan_idx)
                    candidate.idx = current_scan_idx
                    candidate.get_item_fn = datasource.getItem
                    candidate.cardinality = datasource.cardinality
                    candidates.append(candidate)

                    # update the current scan index
                    current_scan_idx += 1

                    # update whether to keep scanning source records
                    keep_scanning_source_records = current_scan_idx < scan_end_idx

                # run operator set on records
                records, record_op_stats_lst, sample_matrix = self.execute_op_set(candidates, op_set)

            # otherwise, process the records in the processing queue for this operator one at a time
            elif len(processing_queues[op_set_id]) > 0:
                candidates = processing_queues[op_set_id]
                records, record_op_stats_lst, sample_matrix = self.execute_op_set(candidates, op_set)

            # update plan stats
            plan_stats.operator_stats[op_set_id].add_record_op_stats(
                record_op_stats_lst,
                source_op_id=prev_op_set_id,
                plan_id=plan.plan_id,
            )

            # add records (which are not filtered) to the cache, if allowed
            if not self.nocache:
                for record in records:
                    if getattr(record, "_passed_filter", True):
                        self.datadir.appendCache(op_set_id, record)

            # update sample matrix
            plan.sample_matrices[op_set_idx] = sample_matrix

            # update processing_queues or output_records
            for record in records:
                if isinstance(op_set[0], FilterOp):
                    if not record._passed_filter:
                        continue
                if next_op_set_id is not None:
                    processing_queues[next_op_set_id].append(record)
                else:
                    output_records.append(record)

            # if we've filtered out all records, terminate early
            if next_op_set_id is not None:
                if processing_queues[next_op_set_id] == []:
                    break

        # if caching was allowed, close the cache
        if not self.nocache:
            for op_set in plan.operator_sets:
                op_set_id = self.compute_op_set_id(op_set)
                self.datadir.closeCache(op_set_id)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats


class PipelinedSingleThreadSentinelPlanExecutor(ExecutionEngine):
    """
    """
    pass


class PipelinedParallelSentinelPlanExecutor(ExecutionEngine):
    """
    This class implements the abstract execute_plan() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute() method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = (
            self.get_parallel_max_workers()
            if self.max_workers is None
            else self.max_workers
        )

    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: Union[DataRecord, List[DataRecord]]):
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        records, record_op_stats_lst = operator(op_input)

        return records, record_op_stats_lst, operator

    def execute_plan(self, plan: SentinelPlan,
                     scan_start_idx: int,
                     scan_end_idx: int,
                     plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        num_samples = scan_end_idx - scan_start_idx
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (records=[{scan_start_idx}:{scan_end_idx}]):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op_set in plan.operator_sets:
            for op in op_set:
                op_id = op.get_op_id()
                op_name = op.op_name()
                op_details = {k: v for k, v in op.get_op_params().items() if k not in ["inputSchema", "outputSchema"]} # TODO
                plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op.op_name(), op_details=op.get_op_params())

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0

        # initialize data structures to help w/processing DAG
        processing_queue = []
        op_id_to_futures_in_flight = {op.get_op_id(): 0 for op in plan.operators}
        op_id_to_operator = {op.get_op_id(): op for op in plan.operators}
        op_id_to_prev_operator = {
            op.get_op_id(): plan.operators[idx - 1] if idx > 0 else None
            for idx, op in enumerate(plan.operators)
        }
        op_id_to_next_operator = {
            op.get_op_id(): plan.operators[idx + 1] if idx + 1 < len(plan.operators) else None
            for idx, op in enumerate(plan.operators)
        }
        op_id_to_op_idx = {op.get_op_id(): idx for idx, op in enumerate(plan.operators)}

        # get handle to DataSource and pre-compute its op_id and size
        source_operator = plan.operators[0]
        source_op_id = source_operator.get_op_id()
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )
        datasource_len = len(datasource)

        # get limit of final limit operator (if one exists)
        final_limit = plan.operators[-1].limit if isinstance(plan.operators[-1], LimitScanOp) else None

        # create thread pool w/max workers
        futures = []
        current_scan_idx = self.scan_start_idx
        with ThreadPoolExecutor(max_workers=plan_workers) as executor:
            # create initial (set of) future(s) to read first source record;
            # construct input DataRecord for DataSourcePhysicalOp
            candidate = DataRecord(schema=SourceRecord, parent_id=None, scan_idx=current_scan_idx)
            candidate.idx = current_scan_idx
            candidate.get_item_fn = datasource.getItem
            candidate.cardinality = datasource.cardinality
            futures.append(executor.submit(PipelinedParallelSentinelPlanExecutor.execute_op_wrapper, source_operator, candidate))
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
                    prev_operator = op_id_to_prev_operator[op_id]
                    plan_stats.operator_stats[op_id].add_record_op_stats(
                        record_op_stats_lst,
                        source_op_id=prev_operator.get_op_id() if prev_operator is not None else None,
                        plan_id=plan.plan_id,
                    )

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
                            candidate = DataRecord(schema=SourceRecord, parent_id=None, scan_idx=current_scan_idx)
                            candidate.idx = current_scan_idx
                            candidate.get_item_fn = datasource.getItem
                            candidate.cardinality = datasource.cardinality
                            new_futures.append(executor.submit(PipelinedParallelSentinelPlanExecutor.execute_op_wrapper, source_operator, candidate))
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
                        future = executor.submit(PipelinedParallelSentinelPlanExecutor.execute_op_wrapper, operator, candidate)
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
                        upstream_op_id_queue = list(filter(lambda tup: tup[0].get_op_id() == upstream_op_id, temp_processing_queue))

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
                        future = executor.submit(PipelinedParallelSentinelPlanExecutor.execute_op_wrapper, operator, candidates)
                        new_futures.append(future)
                        op_id_to_futures_in_flight[operator.get_op_id()] += 1

                    # otherwise, add the candidates back to the processing queue
                    else:
                        processing_queue.extend(candidate_tuples)

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
