from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.corelib import Schema
from palimpzest.elements import DataRecord
from palimpzest.execution import ExecutionEngine
from palimpzest.operators import AggregateOp, DataSourcePhysicalOp, LimitScanOp, MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp
from palimpzest.optimizer import PhysicalPlan

from typing import Union

import time


class SequentialSingleThreadPlanExecutor(ExecutionEngine):
    """
    This class implements the abstract execute_plan() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute() method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute_plan(self, plan: PhysicalPlan,
                     num_samples: Union[int, float] = float("inf"),
                     plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
            print(plan)
            print("---")
            exit(0)

        plan_start_time = time.time()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op in plan.operators:
            op_id = op.get_op_id()
            op_name = op.op_name()
            op_details = {k: str(v) for k, v in op.get_op_params().items()}
            plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op_name, op_details=op_details)

        # initialize list of output records and intermediate variables
        output_records = []
        current_scan_idx = self.scan_start_idx

        # get handle to DataSource and pre-compute its size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )
        datasource_len = len(datasource)

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }

        # execute the plan one operator at a time
        for op_idx, operator in enumerate(plan.operators):
            op_id = operator.get_op_id()
            prev_op_id = (
                plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
            )
            next_op_id = (
                plan.operators[op_idx + 1].get_op_id()
                if op_idx + 1 < len(plan.operators)
                else None
            )

            # initialize output records and record_op_stats for this operator
            records, record_op_stats = [], []

            # invoke datasource operator(s) until we run out of source records or hit the num_samples limit
            if isinstance(operator, DataSourcePhysicalOp):
                keep_scanning_source_records = True
                while keep_scanning_source_records:
                    # construct input DataRecord for DataSourcePhysicalOp
                    # NOTE: this DataRecord will be discarded and replaced by the scan_operator;
                    #       it is simply a vessel to inform the scan_operator which record to fetch
                    candidate = DataRecord(schema=SourceRecord, source_id=current_scan_idx)
                    candidate.idx = current_scan_idx
                    candidate.get_item_fn = datasource.getItem

                    # run DataSourcePhysicalOp on record
                    record_set = operator(candidate)
                    records.extend(record_set.data_records)
                    record_op_stats.extend(record_set.record_op_stats)

                    # update the current scan index
                    current_scan_idx += 1

                    # update whether to keep scanning source records
                    keep_scanning_source_records = (
                        current_scan_idx < datasource_len
                        and len(records) < num_samples
                    )

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
            if not self.nocache:
                for record in records:
                    if getattr(record, "_passed_operator", True):
                        self.datadir.appendCache(operator.targetCacheId, record)

            # update processing_queues or output_records
            for record in records:
                if isinstance(operator, FilterOp):
                    if not record._passed_operator:
                        continue
                if next_op_id is not None:
                    processing_queues[next_op_id].append(record)
                else:
                    output_records.append(record)

            # if we've filtered out all records, terminate early
            if next_op_id is not None:
                if processing_queues[next_op_id] == []:
                    break

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats


class PipelinedSingleThreadPlanExecutor(ExecutionEngine):
    """
    This class implements the abstract execute_plan() method from the ExecutionEngine.
    This class still needs to be sub-classed by another Execution class which implements
    the higher-level execute() method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = 1 if self.max_workers is None else self.max_workers

    def execute_plan(self, plan: PhysicalPlan,
                     num_samples: Union[int, float] = float("inf"),
                     plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op_idx, op in enumerate(plan.operators):
            op_id = op.get_op_id()
            op_name = op.op_name()
            op_details = {k: str(v) for k, v in op.get_op_params().items()}
            plan_stats.operator_stats[op_id] = OperatorStats(op_id=op_id, op_name=op_name, op_details=op_details)

        # initialize list of output records and intermediate variables
        output_records = []
        source_records_scanned = 0
        current_scan_idx = self.scan_start_idx

        # get handle to DataSource and pre-compute its size
        source_operator = plan.operators[0]
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )
        datasource_len = len(datasource)

        # initialize processing queues for each operation
        processing_queues = {
            op.get_op_id(): []
            for op in plan.operators
            if not isinstance(op, DataSourcePhysicalOp)
        }

        # execute the plan until either:
        # 1. all records have been processed, or
        # 2. the final limit operation has completed
        finished_executing, keep_scanning_source_records = False, True
        while not finished_executing:
            for op_idx, operator in enumerate(plan.operators):
                op_id = operator.get_op_id()

                prev_op_id = (
                    plan.operators[op_idx - 1].get_op_id() if op_idx > 1 else None
                )
                next_op_id = (
                    plan.operators[op_idx + 1].get_op_id()
                    if op_idx + 1 < len(plan.operators)
                    else None
                )

                # create empty lists for records and execution stats generated by executing this operator on its next input(s)
                records, record_op_stats = [], []

                # invoke datasource operator(s) until we run out of source records or hit the num_samples limit
                if isinstance(operator, DataSourcePhysicalOp):
                    if keep_scanning_source_records:
                        # construct input DataRecord for DataSourcePhysicalOp
                        # NOTE: this DataRecord will be discarded and replaced by the scan_operator;
                        #       it is simply a vessel to inform the scan_operator which record to fetch
                        candidate = DataRecord(schema=SourceRecord, source_id=current_scan_idx)
                        candidate.idx = current_scan_idx
                        candidate.get_item_fn = datasource.getItem

                        # run DataSourcePhysicalOp on record
                        record_set = operator(candidate)
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
                        # datasources do not have processing queues
                        if isinstance(plan.operators[upstream_op_idx], DataSourcePhysicalOp):
                            continue

                        # check upstream ops which do have a processing queue
                        upstream_op_id = plan.operators[upstream_op_idx].get_op_id()
                        upstream_ops_are_finished = (
                            upstream_ops_are_finished
                            and len(processing_queues[upstream_op_id]) == 0
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
                    if not self.nocache:
                        for record in records:
                            if getattr(record, "_passed_operator", True):
                                self.datadir.appendCache(operator.targetCacheId, record)

                    # update processing_queues or output_records
                    for record in records:
                        if isinstance(operator, FilterOp):
                            if not record._passed_operator:
                                continue
                        if next_op_id is not None:
                            processing_queues[next_op_id].append(record)
                        else:
                            output_records.append(record)

            # update finished_executing based on whether all records have been processed
            still_processing = any([len(queue) > 0 for queue in processing_queues.values()])
            keep_scanning_source_records = (
                current_scan_idx < datasource_len
                and source_records_scanned < num_samples
            )
            finished_executing = not keep_scanning_source_records and not still_processing

            # update finished_executing based on limit
            if isinstance(operator, LimitScanOp):
                finished_executing = (len(output_records) == operator.limit)

        # if caching was allowed, close the cache
        if not self.nocache:
            for operator in plan.operators:
                self.datadir.closeCache(operator.targetCacheId)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return output_records, plan_stats
