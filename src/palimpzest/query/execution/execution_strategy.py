import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import OperatorCostEstimates, PlanStats, RecordOpStats
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.policy import Policy
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import PZSentinelProgressManager

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 num_samples: int | None = None,
                 cache: bool = False,
                 verbose: bool = False,
                 progress: bool = True,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.num_samples = num_samples
        self.cache = cache
        self.verbose = verbose
        self.progress = progress


    def _add_records_to_cache(self, target_cache_id: str, records: list[DataRecord]) -> None:
        """Add each record (which isn't filtered) to the cache for the given target_cache_id."""
        if self.cache:
            for record in records:
                if getattr(record, "passed_operator", True):
                    # self.datadir.append_cache(target_cache_id, record)
                    pass

    def _close_cache(self, target_cache_ids: list[str]) -> None:
        """Close the cache for each of the given target_cache_ids"""
        if self.cache:
            for target_cache_id in target_cache_ids:  # noqa: B007
                # self.datadir.close_cache(target_cache_id)
                pass

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

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, list]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {}
        for op in plan.operators:
            inputs = []
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datareader)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datareader))
                )
                inputs = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            input_queues[op.get_full_op_id()] = inputs

        return input_queues

class SentinelExecutionStrategy(BaseExecutionStrategy, ABC):
    """Base strategy for executing sentinel query plans. Defines how to execute a SentinelPlan."""
    """
    Specialized query processor that implements MAB sentinel strategy
    for coordinating optimization and execution.
    """
    def __init__(
        self,
        val_datasource: DataReader,
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
        self.val_datasource = val_datasource
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.policy = policy
        self.priors = priors
        self.use_final_op_quality = use_final_op_quality
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name

        # special cache which is used for tracking the target record sets for each (source_idx, logical_op_id)
        self.champion_output_cache: dict[int, dict[str, tuple[DataRecordSet, float]]] = {}

        # general cache which maps hash(logical_op_id, phys_op_id, hash(input)) --> record_set
        self.cache: dict[int, DataRecordSet] = {}

        # progress manager used to track progress of the execution
        self.progress_manager: PZSentinelProgressManager | None = None

    def _compute_quality(
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
        physical_op_cls: type[PhysicalOperator],
        source_idx_to_record_sets: dict[int, list[DataRecordSet]],
        source_idx_to_target_record_set: dict[int, DataRecordSet],
    ) -> dict[int, list[DataRecordSet]]:
        """
        NOTE: This approach to cost modeling does not work directly for aggregation queries;
              for these queries, we would ask the user to provide validation data for the step immediately
              before a final aggregation

        NOTE: This function currently assumes that one-to-many converts do NOT create duplicate outputs.
        This assumption would break if, for example, we extracted the breed of every dog in an image.
        If there were two golden retrievers and a bernoodle in an image and we extracted:

            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Bernedoodle"}

        This function would currently give perfect accuracy to the following output:

            {"image": "file1.png", "breed": "Golden Retriever"}
            {"image": "file1.png", "breed": "Bernedoodle"}

        Even though it is missing one of the golden retrievers.
        """
        # extract information about the logical operation performed at this stage of the sentinel plan;
        # NOTE: we can infer these fields from context clues, but in the long-term we should have a more
        #       principled way of getting these directly from attributes either stored in the sentinel_plan
        #       or in the PhysicalOperator
        is_perfect_quality_op = (
            not issubclass(physical_op_cls, LLMConvert)
            and not issubclass(physical_op_cls, LLMFilter)
            and not issubclass(physical_op_cls, RetrieveOp)
        )

        # compute quality of each output computed by this operator
        for source_idx, record_sets in source_idx_to_record_sets.items():
            # if this operation does not involve an LLM, every record_op_stats object gets perfect quality
            if is_perfect_quality_op:
                for record_set in record_sets:
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 1.0
                continue

            # extract target output for this record set
            target_record_set = source_idx_to_target_record_set[source_idx]

            # for each record_set produced by an operation, compute its quality
            for record_set in record_sets:
                record_set = self._compute_quality(physical_op_cls, record_set, target_record_set)

        # return the quality annotated record sets
        return source_idx_to_record_sets

    def _get_target_record_sets(
        self,
        logical_op_id: str,
        source_idx_to_record_set_tuples: dict[int, list[tuple[DataRecordSet, PhysicalOperator, bool]]],
        expected_outputs: dict[int, dict] | None,
    ) -> dict[int, DataRecordSet]:
        # initialize mapping from source index to target record sets
        source_idx_to_target_record_set = {}

        for source_idx, record_set_tuples in source_idx_to_record_set_tuples.items():
            # get the first generated output for this source_idx
            base_target_record = None
            for record_set, _, _ in record_set_tuples:
                if len(record_set) > 0:
                    base_target_record = record_set[0]
                    break

            # compute availability of data
            base_target_present = base_target_record is not None
            labels_present = expected_outputs is not None
            labels_for_source_present = False
            if labels_present and source_idx in expected_outputs:
                labels = expected_outputs[source_idx].get("labels", [])
                labels_dict_lst = labels if isinstance(labels, list) else [labels]
                labels_for_source_present = labels_dict_lst != [] and labels_dict_lst != [None]

            # if we have a base target record and label info, use the label info to construct the target record set
            if base_target_present and labels_for_source_present:
                # get the field_to_score_fn                
                field_to_score_fn = expected_outputs[source_idx].get("score_fn", {})

                # construct the target record set; we force passed_operator to be True for all target records
                target_records = []
                for labels_dict in labels_dict_lst:
                    target_record = base_target_record.copy()
                    for field, value in labels_dict.items():
                        target_record[field] = value
                    target_record.passed_operator = True
                    target_records.append(target_record)

                source_idx_to_target_record_set[source_idx] = DataRecordSet(target_records, None, field_to_score_fn)
                continue

            # get the best computed output for this (source_idx, logical_op_id) so far (if one exists)
            champion_record_set, champion_op_quality = None, None
            if source_idx in self.champion_output_cache and logical_op_id in self.champion_output_cache[source_idx]:
                champion_record_set, champion_op_quality = self.champion_output_cache[source_idx][logical_op_id]

            # get the highest quality output that we just computed
            max_quality_record_set, max_op_quality = self._pick_champion_output(record_set_tuples)

            # if this new output is of higher quality than our previous champion (or if we didn't have
            # a previous champion) then we update our champion record set
            if champion_op_quality is None or (max_op_quality is not None and max_op_quality > champion_op_quality):
                champion_record_set, champion_op_quality = max_quality_record_set, max_op_quality

            # update the cache with the new champion record set and quality
            if source_idx not in self.champion_output_cache:
                self.champion_output_cache[source_idx] = {}
            self.champion_output_cache[source_idx][logical_op_id] = (champion_record_set, champion_op_quality)

            # set the target
            source_idx_to_target_record_set[source_idx] = champion_record_set

        return source_idx_to_target_record_set

    def _pick_champion_output(self, record_set_tuples: list[tuple[DataRecordSet, PhysicalOperator, bool]]) -> tuple[DataRecordSet, float | None]:
        # find the operator with the highest estimated quality and return its record_set
        base_op_cost_est = OperatorCostEstimates(cardinality=1.0, cost_per_record=0.0, time_per_record=0.0, quality=1.0)
        champion_record_set, champion_quality = None, None
        for record_set, op, _ in record_set_tuples:
            # skip failed operations
            if len(record_set) == 0:
                continue

            # get the estimated quality of this operator
            est_quality = op.naive_cost_estimates(base_op_cost_est).quality if self._is_llm_op(op) else 1.0
            if champion_quality is None or est_quality > champion_quality:
                champion_record_set, champion_quality = record_set, est_quality

        return champion_record_set, champion_quality

    def _flatten_record_sets(self, source_idx_to_record_sets: dict[int, list[DataRecordSet]]) -> tuple[list[DataRecord], list[RecordOpStats]]:
        """
        Flatten the list of record sets and record op stats for each source_idx.
        """
        all_records, all_record_op_stats = [], []
        for _, record_sets in source_idx_to_record_sets.items():
            for record_set in record_sets:
                all_records.extend(record_set.data_records)
                all_record_op_stats.extend(record_set.record_op_stats)

        return all_records, all_record_op_stats

    def _execute_op_set(self, op_input_pairs: list[tuple[PhysicalOperator, DataRecord | int]]) -> tuple[dict[int, list[tuple[DataRecordSet, PhysicalOperator, bool]]], dict[str, int]]:
        def execute_op_wrapper(operator, input) -> tuple[DataRecordSet, PhysicalOperator, DataRecord | int]:
            record_set = operator(input)
            return record_set, operator, input

        # TODO: modify unit tests to always have record_op_stats so we can use record_op_stats for source_idx
        # for scan operators, `input` will be the source_idx
        def get_source_idx(input):
            return input.source_idx if isinstance(input, DataRecord) else input

        def get_hash(operator, input):
            return hash(f"{operator.get_full_op_id()}{hash(input)}")

        # initialize mapping from source indices to output record sets
        source_idx_to_record_sets_and_ops = {get_source_idx(input): [] for _, input in op_input_pairs}

        # if any operations were previously executed, read the results from the cache
        final_op_input_pairs = []
        for operator, input in op_input_pairs:
            # compute hash
            op_input_hash = get_hash(operator, input)

            # get result from cache
            if op_input_hash in self.cache:
                source_idx = get_source_idx(input)
                record_set, operator = self.cache[op_input_hash]
                source_idx_to_record_sets_and_ops[source_idx].append((record_set, operator, False))

            # otherwise, add to final_op_input_pairs
            else:
                final_op_input_pairs.append((operator, input))

        # keep track of the number of llm operations
        num_llm_ops = 0

        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = [
                executor.submit(execute_op_wrapper, operator, input)
                for operator, input in final_op_input_pairs
            ]
            output_record_sets = []
            while len(futures) > 0:
                done_futures, not_done_futures = wait(futures, timeout=PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS)
                for future in done_futures:
                    # update output record sets
                    record_set, operator, input = future.result()
                    output_record_sets.append((record_set, operator, input))

                    # update cache
                    op_input_hash = get_hash(operator, input)
                    self.cache[op_input_hash] = (record_set, operator)

                    # update progress manager
                    if self._is_llm_op(operator):
                        num_llm_ops += 1
                        self.progress_manager.incr(operator.get_logical_op_id(), num_samples=1, total_cost=record_set.get_total_cost())

                # update futures
                futures = list(not_done_futures)

            # update mapping from source_idx to record sets and operators
            for record_set, operator, input in output_record_sets:
                # get the source_idx associated with this input record;
                source_idx = get_source_idx(input)

                # add record_set to mapping from source_idx --> record_sets
                source_idx_to_record_sets_and_ops[source_idx].append((record_set, operator, True))

        return source_idx_to_record_sets_and_ops, num_llm_ops

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_retrieve = isinstance(physical_op, RetrieveOp) and isinstance(physical_op.index, Collection)
        return is_llm_convert or is_llm_filter or is_llm_retrieve

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan, expected_outputs: dict[str, dict]):
        """Execute a SentinelPlan according to strategy"""
        pass
