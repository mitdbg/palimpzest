import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS, Cardinality
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import PlanStats, SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import PZSentinelProgressManager
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 num_samples: int | None = None,
                 verbose: bool = False,
                 progress: bool = True,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.num_samples = num_samples
        self.verbose = verbose
        self.progress = progress


class ExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing query plans. Defines how to execute a PhysicalPlan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(self, plan: PhysicalPlan) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, dict[str, list]]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {f"{topo_idx}-{op.get_full_op_id()}": {} for topo_idx, op in enumerate(plan)}
        for topo_idx, op in enumerate(plan):
            full_op_id = op.get_full_op_id()
            unique_op_id = f"{topo_idx}-{full_op_id}"
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datasource)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datasource))
                )
                input_queues[unique_op_id][f"source_{full_op_id}"] = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            elif isinstance(op, ContextScanOp):
                input_queues[unique_op_id][f"source_{full_op_id}"] = [None]
            else:
                for source_unique_full_op_id in plan.get_source_unique_full_op_ids(topo_idx, op):
                    input_queues[unique_op_id][source_unique_full_op_id] = []

        return input_queues

class SentinelExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing sentinel query plans. Defines how to execute a SentinelPlan."""
    """
    Specialized query processor that implements MAB sentinel strategy
    for coordinating optimization and execution.
    """
    def __init__(
        self,
        k: int,
        j: int,
        sample_budget: int,
        policy: Policy,
        priors: dict | None = None,
        use_final_op_quality: bool = False,
        seed: int = 42,
        exp_name: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.policy = policy
        self.priors = priors
        self.use_final_op_quality = use_final_op_quality
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name

        # general cache which maps hash(logical_op_id, phys_op_id, hash(input)) --> record_set
        self.cache: dict[int, DataRecordSet] = {}

        # progress manager used to track progress of the execution
        self.progress_manager: PZSentinelProgressManager | None = None

    def _old_compute_quality(
            self,
            physical_op_cls: type[PhysicalOperator],
            record_set: DataRecordSet,
            target_record_set: DataRecordSet,
        ) -> DataRecordSet:
        """
        Compute the quality for the given `record_set` by comparing it to the `target_record_set`.

        Update the record_set by assigning the quality to each entry in its record_op_stats and
        returning the updated record_set.
        """
        # if this operation failed
        if len(record_set) == 0:
            record_set.record_op_stats[0].quality = 0.0

        # if this operation is a filter:
        # - return 1.0 if there's a match in the expected output which this operator does not filter out and 0.0 otherwise
        elif issubclass(physical_op_cls, FilterOp):
            # NOTE: we know that record_set.data_records will contain a single entry for a filter op
            record = record_set.data_records[0]

            # search for a record in the target with the same set of fields
            found_match_in_target = False
            for target_record in target_record_set:
                all_correct = True
                for field, value in record.field_values.items():
                    if value != target_record[field]:
                        all_correct = False
                        break

                if all_correct:
                    found_match_in_target = target_record.passed_operator
                    break

            # set quality based on whether we found a match in the target and return
            record_set.record_op_stats[0].quality = int(record.passed_operator == found_match_in_target)

            return record_set

        # if this is a successful convert operation
        else:
            # NOTE: the following computation assumes we do not project out computed values
            #       (and that the validation examples provide all computed fields); even if
            #       a user program does add projection, we can ignore the projection on the
            #       validation dataset and use the champion model (as opposed to the validation
            #       output) for scoring fields which have their values projected out

            # GREEDY ALGORITHM
            # for each record in the expected output, we look for the computed record which maximizes the quality metric;
            # once we've identified that computed record we remove it from consideration for the next expected output
            field_to_score_fn = target_record_set.get_field_to_score_fn()
            for target_record in target_record_set:
                best_quality, best_record_op_stats = 0.0, None
                for record_op_stats in record_set.record_op_stats:
                    # if we already assigned this record a quality, skip it
                    if record_op_stats.quality is not None:
                        continue

                    # compute number of matches between this record's computed fields and this expected record's outputs
                    total_quality = 0
                    for field in record_op_stats.generated_fields:
                        computed_value = record_op_stats.record_state.get(field, None)
                        expected_value = target_record[field]

                        # get the metric function for this field
                        score_fn = field_to_score_fn.get(field, "exact")

                        # compute exact match
                        if score_fn == "exact":
                            total_quality += int(computed_value == expected_value)

                        # compute UDF metric
                        elif callable(score_fn):
                            total_quality += score_fn(computed_value, expected_value)

                        # otherwise, throw an exception
                        else:
                            raise Exception(f"Unrecognized score_fn: {score_fn}")

                    # compute recall and update best seen so far
                    quality = total_quality / len(record_op_stats.generated_fields)
                    if quality > best_quality:
                        best_quality = quality
                        best_record_op_stats = record_op_stats

                # set best_quality as quality for the best_record_op_stats
                if best_record_op_stats is not None:
                    best_record_op_stats.quality = best_quality

        # for any records which did not receive a quality, set it to 0.0 as these are unexpected extras
        for record_op_stats in record_set.record_op_stats:
            if record_op_stats.quality is None:
                record_op_stats.quality = 0.0

        return record_set

    def _score_quality(
        self,
        validator: Validator,
        source_indices_to_record_sets: dict[tuple[str], list[tuple[DataRecordSet, PhysicalOperator]]],
    ) -> dict[int, list[DataRecordSet]]:
        # extract information about the logical operation performed at this stage of the sentinel plan;
        # NOTE: we can infer these fields from context clues, but in the long-term we should have a more
        #       principled way of getting these directly from attributes either stored in the sentinel_plan
        #       or in the PhysicalOperator
        def is_perfect_quality_op(op: PhysicalOperator):
            return (
                not isinstance(op, LLMConvert)
                and not isinstance(op, LLMFilter)
                and not isinstance(op, RetrieveOp)
                and not isinstance(op, JoinOp)
            )

        # compute quality of each output computed by this operator
        for _, record_set_tuples in source_indices_to_record_sets.items():
            for record_set, op in record_set_tuples:
                # if this operation does not involve an LLM, every record_op_stats object gets perfect quality
                if is_perfect_quality_op(op):
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 1.0
                    continue

                # if the operation failed, assign 0.0 quality
                if len(record_set) == 0:
                    record_set.record_op_stats[0].quality = 0.0
                    continue

                # for each record_set produced by an operation, compute its quality
                if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                    fields = op.generated_fields
                    input_record = record_set.input
                    output = record_set.data_records[0].to_dict(project_cols=fields)
                    record_set.record_op_stats[0].quality = validator._score_map(op, fields, input_record, output)

                elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                    fields = op.generated_fields
                    input_record = record_set.input
                    output = []
                    for data_record in record_set.data_records:
                        output.append(data_record.to_dict(project_cols=fields))
                    score = validator._score_flat_map(op, fields, input_record, output)
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = score

                elif isinstance(op, LLMFilter):
                    filter_str = op.filter_obj.filter_condition
                    input_record = record_set.input
                    output = record_set.data_records[0].passed_operator
                    record_set.record_op_stats[0].quality = validator._score_filter(op, filter_str, input_record, output)

                elif isinstance(op, JoinOp):
                    condition = op.condition
                    for left_idx, left_input_record in enumerate(record_set.input[0]):
                        for right_idx, right_input_record in enumerate(record_set.input[1]):
                            record_idx = left_idx * len(record_set.input[0]) + right_idx
                            output = record_set.data_records[record_idx].passed_operator
                            record_set.record_op_stats[record_idx].quality = validator._score_join(op, condition, left_input_record, right_input_record, output)

        # return the quality annotated record sets
        return source_indices_to_record_sets

    def _execute_op_set(self, unique_logical_op_id: str, op_inputs: list[tuple[PhysicalOperator, str | tuple, int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]]]) -> tuple[dict[int, list[tuple[DataRecordSet, PhysicalOperator, bool]]], dict[str, int]]:
        def execute_op_wrapper(operator: PhysicalOperator, source_indices: str | tuple, input: int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]) -> tuple[DataRecordSet, PhysicalOperator, list[DataRecord] | list[int]]:
            # operator is a join
            record_set = operator(input[0], input[1]) if isinstance(operator, JoinOp) else operator(input)
            return record_set, operator, source_indices, input

        def get_hash(operator: PhysicalOperator, input: int | DataRecord | list[DataRecord] | tuple[list[DataRecord]]):
            if isinstance(input, list):
                input = tuple(input)
            elif isinstance(input, tuple):
                input = (tuple(input[0]), tuple(input[1]))
            return hash(f"{operator.get_full_op_id()}{hash(input)}")

        # initialize mapping from source indices to output record sets
        source_indices_to_record_sets_and_ops = {source_indices: [] for _, source_indices, _ in op_inputs}

        # if any operations were previously executed, read the results from the cache
        final_op_inputs = []
        for operator, source_indices, input in op_inputs:
            # compute hash
            op_input_hash = get_hash(operator, input)

            # get result from cache
            if op_input_hash in self.cache:
                record_set, operator = self.cache[op_input_hash]
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, False))

            # otherwise, add to final_op_inputs
            else:
                final_op_inputs.append((operator, source_indices, input))

        # keep track of the number of llm operations
        num_llm_ops = 0

        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = [
                executor.submit(execute_op_wrapper, operator, source_indices, input)
                for operator, source_indices, input in final_op_inputs
            ]
            output_record_sets = []
            while len(futures) > 0:
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)
                for future in done_futures:
                    # update output record sets
                    record_set, operator, source_indices, input = future.result()
                    output_record_sets.append((record_set, operator, source_indices, input))

                    # update cache
                    op_input_hash = get_hash(operator, input)
                    self.cache[op_input_hash] = (record_set, operator)

                    # update progress manager
                    if self._is_llm_op(operator):
                        num_llm_ops += 1
                        self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=record_set.get_total_cost())

                # update futures
                futures = list(not_done_futures)

            # update mapping from source_indices to record sets and operators
            for record_set, operator, source_indices, input in output_record_sets:
                # add record_set to mapping from source_indices --> record_sets
                record_set.input = input
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, True))

        return source_indices_to_record_sets_and_ops, num_llm_ops

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_retrieve = isinstance(physical_op, RetrieveOp) and isinstance(physical_op.index, Collection)
        is_llm_join = isinstance(physical_op, JoinOp)
        return is_llm_convert or is_llm_filter or is_llm_retrieve or is_llm_join

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator) -> SentinelPlanStats:
        """Execute a SentinelPlan according to strategy"""
        pass
