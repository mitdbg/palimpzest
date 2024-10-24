from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS, PickOutputStrategy
from palimpzest.corelib.schemas import SourceRecord
from palimpzest.dataclasses import OperatorStats, PlanStats
from palimpzest.elements import DataRecord, DataRecordSet
from palimpzest.execution import ExecutionEngine, SequentialSingleThreadPlanExecutor
from palimpzest.operators import *
from palimpzest.optimizer import SentinelPlan
from palimpzest.utils import create_sample_mask, getChampionModel

from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from typing import List, Tuple, Union

import numpy as np

import time

# NOTE: I would like for this to be a util, but its reliance on operator specifics
#       makes it impossible to put in utils (which multiple operator files import from)
def getChampionConvertOperator(operators):
    """
    NOTE: this function is one place where inductive bias (in the form of our prior belief
    about the best operator(s)) is inserted into the PZ optimizer.

    If we want to support a fully non-inductive bias optimizer, which also works w/out
    the presence of validation data, then we would need to have the user specify their
    preferred champion operator (or give a ranking of operators).

    Thus, we can think of this as being a quick and dirty way for us to provide that
    preference / ranking information (rather than passing it all the way through the executor).
    """
    convert_operator_class_ranking = {
        LLMConvertConventional: 4,
        LLMConvertBonded: 3,
        # setting both token reduced converts to same ranking
        TokenReducedConvertConventional: 2,
        TokenReducedConvertBonded: 2,
        # setting all code synthesis converts to same ranking
        CodeSynthesisConvertNone: 1,
        CodeSynthesisConvertSingle: 1,
        CodeSynthesisConvertExampleEnsemble: 1,
        CodeSynthesisConvertAdviceEnsemble: 1,
        CodeSynthesisConvertAdviceEnsembleValidation: 1,
    }
    filter_operator_class_ranking = {
        LLMFilter: 1,
    }
    model_ranking = {
        Model.GPT_4o: 4,
        Model.GPT_4o_V: 4,
        Model.GPT_4o_MINI: 3,
        Model.GPT_4o_MINI_V: 3,
        Model.MIXTRAL: 2,
        Model.LLAMA3: 1,
        Model.LLAMA3_V: 1,
    }

    # compute product of operator and model ranking
    champion_convert_operator, champion_product = None, 0
    for operator in operators:
        operator_ranking = (
            convert_operator_class_ranking[operator.__class__]
            if isinstance(operator, ConvertOp)
            else filter_operator_class_ranking[operator.__class__]
        )
        product = (
            operator_ranking * model_ranking[operator.model]
            if not isinstance(operator, CodeSynthesisConvert)
            else operator_ranking * 1.0
        )
        if product > champion_product:
            champion_convert_operator = operator

    return champion_convert_operator


class SequentialSingleThreadSentinelPlanExecutor(SequentialSingleThreadPlanExecutor):
    """
    This class sub-classes the SequentialSingleThreadPlanExecutor (which means it inherits
    that classes execute_plan() method).

    This class then adds the execute_sentinel_plan() method, which is used by the SentinelExecution
    engine(s) to execute sentinel plans prior to query optimization.

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

    def pick_champion_output(self, op_set_record_sets: List[Tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        """
        NOTE: this function is one place where inductive bias (in the form of our prior belief
        about the best operator(s)) is inserted into the PZ optimizer.
        """
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # get the preferred convert operator
        operators = set(map(lambda tup: tup[1], op_set_record_sets))
        champion_convert_operator = getChampionConvertOperator(operators)

        # get the champion model(s) (for filters)
        champion_model = getChampionModel(self.available_models)
        champion_vision_model = getChampionModel(self.available_models, vision=True)

        # return output from the bonded query or filter with the best model;
        # ignore response from all other operations
        champion_record_set = None
        for record_set, operator in op_set_record_sets:
            if operator == champion_convert_operator:
                champion_record_set = record_set
                break

            if isinstance(operator, LLMFilter) and operator.model in [champion_model, champion_vision_model]:
                champion_record_set = record_set
                break

        return champion_record_set

    def pick_ensemble_output(self, op_set_record_sets: List[Tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # NOTE: I don't like that this assumes the models are consistent in
        #       how they order their record outputs for one-to-many converts;
        #       eventually we can try out more robust schemes to account for
        #       differences in ordering
        # aggregate records at each index in the response
        idx_to_records = {}
        for record_set, _ in op_set_record_sets:
            for idx, record in enumerate(record_set):
                if idx not in idx_to_records:
                    idx_to_records[idx] = [record]
                else:
                    idx_to_records.append(record)

        # compute most common answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records = idx_to_records[idx]
            most_common_record = max(set(records), key=records.count)
            out_records.append(most_common_record)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, [])

    def execute_op_set(
            self,
            input_records: List[DataRecord],
            op_set: List[PhysicalOperator],
            sample_matrix: np.ndarray,
        ) -> Tuple[Dict[str, List[DataRecordSet]], Dict[str, DataRecordSet]]:
        # handle aggregate operators
        if isinstance(op_set[0], AggregateOp):
            # NOTE: will need to change this if we ever have competing aggregate implemenations
            operator = op_set[0]
            record_set = operator(input_records)

            # NOTE: the convention for now is to use the source_id of the final input record
            #       (both for computing source_ids and parent_ids; in the future we will support
            #       having multiple parent / source ids)
            source_id = input_records[-1]._source_id
            all_record_sets[source_id] = [record_set]
            champion_record_sets[source_id] = record_set

            return all_record_sets, champion_record_sets

        # handle limit; there is only a single implementation of a limit
        if isinstance(op_set[0], LimitScanOp):
            limit_op = op_set[0]
            all_record_sets, champion_record_sets = {}, {}
            for record in input_records:
                record_set = limit_op(record)
                all_record_sets[record._source_id] = [record_set]
                champion_record_sets[record._source_id] = record_set

                # NOTE: if suffices to check the length of all_record_sets because the implementation
                #       of the LimitScanOp takes in (and outputs) a single record at a time
                if len(all_record_sets) == limit_op.limit:
                    return all_record_sets, champion_record_sets

            # if there are fewer records than the limit, return what we have
            return all_record_sets, champion_record_sets

        # run operator set on input records
        all_record_sets, champion_record_sets = {}, {}
        for record_idx, record in enumerate(input_records):
            # run each (sampled) operator in the op_set on the input record
            op_set_record_sets = []
            for op_idx, operator in enumerate(op_set):
                if sample_matrix[record_idx, op_idx]:
                    # run operator on candidate record
                    record_set = operator(record)

                    # add record_set to list of record_sets computed for this op_set by different operators
                    op_set_record_sets.append((record_set, operator))

            # select the champion (i.e. best) record_set from all the record sets computed for this operator
            champion_record_set = self.pick_output_fn(op_set_record_sets)

            # get the source_id associated with this input record
            source_id = record._source_id

            # add champion record_set to mapping from source_id --> champion record_set
            champion_record_sets[source_id] = champion_record_set

            # add all record_sets computed for this source_id to mapping from source_id --> record_sets
            all_record_sets[source_id] = [tup[0] for tup in op_set_record_sets]

        return all_record_sets, champion_record_sets


    def execute_sentinel_plan(self, plan: SentinelPlan, num_samples: int, plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op_set in plan.operator_sets:
            op_set_id = SentinelPlan.compute_op_set_id(op_set)
            op_set_str = ",".join([op.op_name() for op in op_set])
            op_set_name = f"OpSet({op_set_str})"
            op_set_details = {
                op.op_name(): {k: str(v) for k, v in op.get_op_params().items()}
                for op in op_set
            }
            plan_stats.operator_stats[op_set_id] = OperatorStats(op_id=op_set_id, op_name=op_set_name, op_details=op_set_details)

        # get handle to DataSource (# making the assumption that first operator_set can only be a scan
        source_operator = plan.operator_sets[0][0]
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )

        # initialize output variables
        all_outputs, champion_outputs = {}, {}

        # create initial set of candidates for source scan operator
        candidates = []
        for sample_idx in range(num_samples):
            candidate = DataRecord(schema=SourceRecord, source_id=sample_idx)
            candidate.idx = sample_idx
            candidate.get_item_fn = partial(datasource.getItem, val=True)
            candidates.append(candidate)

        # NOTE: because we need to dynamically create sample matrices for each operator,
        #       sentinel execution must be executed one operator at a time (i.e. sequentially)
        # execute operator sets in sequence
        for op_set_idx, op_set in enumerate(plan.operator_sets):
            op_set_id = SentinelPlan.compute_op_set_id(op_set)
            prev_op_set_id = (
                SentinelPlan.compute_op_set_id(plan.operator_sets[op_set_idx - 1])
                if op_set_idx > 1
                else None
            )
            next_op_set_id = (
                SentinelPlan.compute_op_set_id(plan.operator_sets[op_set_idx + 1])
                if op_set_idx + 1 < len(plan.operator_sets)
                else None
            )

            # create a sample matrix
            sample_matrix, record_to_row_map, phys_op_to_col_map = create_sample_mask(candidates, op_set, self.rank)

            # run operator set on records
            source_id_to_record_sets, source_id_to_champion_record_set = self.execute_op_set(candidates, op_set, sample_matrix)
            
            # for scan operators, we need to correct the source_id to match what is provided by the DataSource.getItem() method
            if isinstance(op_set[0], DataSourcePhysicalOp):
                new_source_id_to_record_sets, new_source_id_to_champion_record_set, new_record_to_row_map = {}, {}, {}
                for source_id, record_sets in source_id_to_record_sets.items():
                    champion_record_set = source_id_to_champion_record_set[source_id]
                    row_map = record_to_row_map[source_id]

                    # update mapping to use new source_id
                    new_source_id = record_sets[0].source_id
                    new_source_id_to_record_sets[new_source_id] = record_sets
                    new_source_id_to_champion_record_set[new_source_id] = champion_record_set
                    new_record_to_row_map[new_source_id] = row_map

                source_id_to_record_sets = new_source_id_to_record_sets
                source_id_to_champion_record_set = new_source_id_to_champion_record_set
                record_to_row_map = new_record_to_row_map

            # set sample_matrices information for sentinel plan
            plan.sample_matrices[op_set_id] = (sample_matrix, record_to_row_map, phys_op_to_col_map)

            # update all_outputs and champion_outputs dictionary
            all_outputs[op_set_id] = source_id_to_record_sets
            champion_outputs[op_set_id] = source_id_to_champion_record_set

            # flatten lists of records and record_op_stats
            all_records, all_record_op_stats = [], []
            for _, record_sets in source_id_to_record_sets.items():
                for record_set in record_sets:
                    all_records.extend(record_set.data_records)
                    all_record_op_stats.extend(record_set.record_op_stats)

            # update plan stats
            plan_stats.operator_stats[op_set_id].add_record_op_stats(
                all_record_op_stats,
                source_op_id=prev_op_set_id,
                plan_id=plan.plan_id,
            )

            # add records (which are not filtered) to the cache, if allowed
            if not self.nocache:
                for record in all_records:
                    if getattr(record, "_passed_filter", True):
                        self.datadir.appendCache(op_set_id, record)

            # update candidates for next operator; we use champion outputs as input
            candidates = []
            if next_op_set_id is not None:
                for _, record_set in champion_outputs[op_set_id].items():
                    for record in record_set:
                        if isinstance(op_set[0], FilterOp):
                            if not record._passed_filter:
                                continue
                        candidates.append(record)

            # if we've filtered out all records, terminate early
            if next_op_set_id is not None and candidates == []:
                break

        # if caching was allowed, close the cache
        if not self.nocache:
            for op_set in plan.operator_sets:
                op_set_id = SentinelPlan.compute_op_set_id(op_set)
                self.datadir.closeCache(op_set_id)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return all_outputs, champion_outputs, plan_stats


class SequentialParallelSentinelPlanExecutor(SequentialSingleThreadPlanExecutor):
    """
    TODO: change after SIGMOD
    NOTE: This class inherits from SequentialSingleThreadPlanExecutor which means
          its execute_plan() method will still be sequential.
    """
    def __init__(self, pick_output_strategy: PickOutputStrategy = PickOutputStrategy.CHAMPION, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.max_workers = self.get_parallel_max_workers()
        # TODO: undo
        self.max_workers = 4
        self.pick_output_fn = (
            self.pick_champion_output
            if pick_output_strategy == PickOutputStrategy.CHAMPION
            else self.pick_ensemble_output
        )

    # TODO: factor out into shared base class
    def pick_champion_output(self, op_set_record_sets: List[Tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        """
        NOTE: this function is one place where inductive bias (in the form of our prior belief
        about the best operator(s)) is inserted into the PZ optimizer.
        """
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # get the preferred convert operator
        operators = set(map(lambda tup: tup[1], op_set_record_sets))
        champion_convert_operator = getChampionConvertOperator(operators)

        # get the champion model(s) (for filters)
        champion_model = getChampionModel(self.available_models)
        champion_vision_model = getChampionModel(self.available_models, vision=True)

        # return output from the bonded query or filter with the best model;
        # ignore response from all other operations
        champion_record_set = None
        for record_set, operator in op_set_record_sets:
            if operator == champion_convert_operator:
                champion_record_set = record_set
                break

            if isinstance(operator, LLMFilter) and operator.model in [champion_model, champion_vision_model]:
                champion_record_set = record_set
                break

        return champion_record_set

    # TODO: factor out into shared base class
    def pick_ensemble_output(self, op_set_record_sets: List[Tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # NOTE: I don't like that this assumes the models are consistent in
        #       how they order their record outputs for one-to-many converts;
        #       eventually we can try out more robust schemes to account for
        #       differences in ordering
        # aggregate records at each index in the response
        idx_to_records = {}
        for record_set, _ in op_set_record_sets:
            for idx, record in enumerate(record_set):
                if idx not in idx_to_records:
                    idx_to_records[idx] = [record]
                else:
                    idx_to_records.append(record)

        # compute most common answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records = idx_to_records[idx]
            most_common_record = max(set(records), key=records.count)
            out_records.append(most_common_record)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, [])

    @staticmethod
    def execute_op_wrapper(operator: PhysicalOperator, op_input: Union[DataRecord, List[DataRecord]]) -> Tuple[DataRecordSet, PhysicalOperator]:
        """
        Wrapper function around operator execution which also and returns the operator.
        This is useful in the parallel setting(s) where operators are executed by a worker pool,
        and it is convenient to return the op_id along with the computation result.
        """
        record_set = operator(op_input)

        return record_set, operator, op_input

    def execute_op_set(self, candidates, op_set, sample_matrix):
        # TODO: post-SIGMOD we will need to modify this to:
        # - submit all candidates for aggregate operators
        # - handle limits
        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # for non-zero entries in sample matrix: create future task
            futures = []
            for row in range(sample_matrix.shape[0]):
                for col in range(sample_matrix.shape[1]):
                    if sample_matrix[row, col] > 0:
                        candidate = candidates[row]
                        operator = op_set[col]
                        future = executor.submit(SequentialParallelSentinelPlanExecutor.execute_op_wrapper, operator, candidate)
                        futures.append(future)

            # compute output record_set for each (operator, candidate) pair
            output_record_sets = []
            while len(futures) > 0:
                # get the set of futures that have (and have not) finished in the last PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)

                # cast not_done_futures from a set to a list so we can append to it
                not_done_futures = list(not_done_futures)

                # process finished futures
                for future in done_futures:
                    # get the result and add it to the 
                    record_set, operator, candidate = future.result()
                    output_record_sets.append((record_set, operator, candidate))

                # update list of futures
                futures = not_done_futures

            # computing mapping from source_id to record sets for all operators and for champion operator
            all_record_sets, champion_record_sets = {}, {}
            for candidate in candidates:
                op_set_record_sets = []
                for record_set, operator, candidate_ in output_record_sets:
                    if candidate == candidate_:
                        op_set_record_sets.append((record_set, operator))
                
                # select the champion (i.e. best) record_set from all the record sets computed for this operator
                champion_record_set = self.pick_output_fn(op_set_record_sets)

                # get the source_id associated with this input record
                source_id = candidate._source_id

                # add champion record_set to mapping from source_id --> champion record_set
                champion_record_sets[source_id] = champion_record_set

                # add all record_sets computed for this source_id to mapping from source_id --> record_sets
                all_record_sets[source_id] = [tup[0] for tup in op_set_record_sets]

        return all_record_sets, champion_record_sets


    def execute_sentinel_plan(self, plan: SentinelPlan, num_samples: int, plan_workers: int = 1):
        """Initialize the stats and the execute the plan."""
        if self.verbose:
            print("----------------------")
            print(f"PLAN[{plan.plan_id}] (n={num_samples}):")
            print(plan)
            print("---")

        plan_start_time = time.time()

        # initialize plan stats and operator stats
        plan_stats = PlanStats(plan_id=plan.plan_id, plan_str=str(plan))
        for op_set in plan.operator_sets:
            op_set_id = SentinelPlan.compute_op_set_id(op_set)
            op_set_str = ",".join([op.op_name() for op in op_set])
            op_set_name = f"OpSet({op_set_str})"
            op_set_details = {
                op.op_name(): {k: str(v) for k, v in op.get_op_params().items()}
                for op in op_set
            }
            plan_stats.operator_stats[op_set_id] = OperatorStats(op_id=op_set_id, op_name=op_set_name, op_details=op_set_details)

        # get handle to DataSource (# making the assumption that first operator_set can only be a scan
        source_operator = plan.operator_sets[0][0]
        datasource = (
            self.datadir.getRegisteredDataset(source_operator.dataset_id)
            if isinstance(source_operator, MarshalAndScanDataOp)
            else self.datadir.getCachedResult(source_operator.dataset_id)
        )

        # initialize output variables
        all_outputs, champion_outputs = {}, {}
        
        # create initial set of candidates for source scan operator
        candidates = []
        for sample_idx in range(num_samples):
            candidate = DataRecord(schema=SourceRecord, source_id=sample_idx)
            candidate.idx = sample_idx
            candidate.get_item_fn = partial(datasource.getItem, val=True)
            candidates.append(candidate)

        # NOTE: because we need to dynamically create sample matrices for each operator,
        #       sentinel execition must be executed one operator at a time (i.e. sequentially)
        # execute operator sets in sequence
        for op_set_idx, op_set in enumerate(plan.operator_sets):
            op_set_id = SentinelPlan.compute_op_set_id(op_set)
            prev_op_set_id = (
                SentinelPlan.compute_op_set_id(plan.operator_sets[op_set_idx - 1])
                if op_set_idx > 1
                else None
            )
            next_op_set_id = (
                SentinelPlan.compute_op_set_id(plan.operator_sets[op_set_idx + 1])
                if op_set_idx + 1 < len(plan.operator_sets)
                else None
            )

            # create a sample matrix
            sample_matrix, record_to_row_map, phys_op_to_col_map = create_sample_mask(candidates, op_set, self.rank)

            # run operator set on records
            source_id_to_record_sets, source_id_to_champion_record_set = self.execute_op_set(candidates, op_set, sample_matrix)
            
            # for scan operators, we need to correct the source_id to match what is provided by the DataSource.getItem() method
            if isinstance(op_set[0], DataSourcePhysicalOp):
                new_source_id_to_record_sets, new_source_id_to_champion_record_set, new_record_to_row_map = {}, {}, {}
                for source_id, record_sets in source_id_to_record_sets.items():
                    champion_record_set = source_id_to_champion_record_set[source_id]
                    row_map = record_to_row_map[source_id]

                    # update mapping to use new source_id
                    new_source_id = record_sets[0].source_id
                    new_source_id_to_record_sets[new_source_id] = record_sets
                    new_source_id_to_champion_record_set[new_source_id] = champion_record_set
                    new_record_to_row_map[new_source_id] = row_map

                source_id_to_record_sets = new_source_id_to_record_sets
                source_id_to_champion_record_set = new_source_id_to_champion_record_set
                record_to_row_map = new_record_to_row_map

            # set sample_matrices information for sentinel plan
            plan.sample_matrices[op_set_id] = (sample_matrix, record_to_row_map, phys_op_to_col_map)

            # update all_outputs and champion_outputs dictionary
            all_outputs[op_set_id] = source_id_to_record_sets
            champion_outputs[op_set_id] = source_id_to_champion_record_set

            # flatten lists of records and record_op_stats
            all_records, all_record_op_stats = [], []
            for _, record_sets in source_id_to_record_sets.items():
                for record_set in record_sets:
                    all_records.extend(record_set.data_records)
                    all_record_op_stats.extend(record_set.record_op_stats)

            # update plan stats
            plan_stats.operator_stats[op_set_id].add_record_op_stats(
                all_record_op_stats,
                source_op_id=prev_op_set_id,
                plan_id=plan.plan_id,
            )

            # add records (which are not filtered) to the cache, if allowed
            if not self.nocache:
                for record in all_records:
                    if getattr(record, "_passed_filter", True):
                        self.datadir.appendCache(op_set_id, record)

            # update candidates for next operator; we use champion outputs as input
            candidates = []
            if next_op_set_id is not None:
                for _, record_set in champion_outputs[op_set_id].items():
                    for record in record_set:
                        if isinstance(op_set[0], FilterOp):
                            if not record._passed_filter:
                                continue
                        candidates.append(record)

            # if we've filtered out all records, terminate early
            if next_op_set_id is not None and candidates == []:
                break

        # if caching was allowed, close the cache
        if not self.nocache:
            for op_set in plan.operator_sets:
                op_set_id = SentinelPlan.compute_op_set_id(op_set)
                self.datadir.closeCache(op_set_id)

        # finalize plan stats
        total_plan_time = time.time() - plan_start_time
        plan_stats.finalize(total_plan_time)

        return all_outputs, champion_outputs, plan_stats
