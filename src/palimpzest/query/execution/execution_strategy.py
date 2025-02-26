import logging
from abc import ABC, abstractmethod

import numpy as np

from palimpzest.core.data.dataclasses import OperatorCostEstimates, PlanStats, RecordOpStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.tools.logger import setup_logger

logger = setup_logger(__name__)


class ExecutionStrategy(ABC):
    """Base strategy for executing query plans. Defines how to execute a PhysicalPlan.
    """
    def __init__(self, 
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 num_samples: int | None = None,
                 cache: bool = False,
                 verbose: bool = False,
                 progress: bool = True,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.num_samples = num_samples
        self.cache = cache
        self.verbose = verbose
        self.progress = progress
        logger.pz_logger.set_console_level(logging.DEBUG if verbose else logging.ERROR)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(
        self,
        plan: PhysicalPlan,
        num_samples: int | float = float("inf"),
        workers: int = 1
    ) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass

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
            input_queues[op.get_op_id()] = inputs

        return input_queues

class SentinelExecutionStrategy(ABC):
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
            sample_all_ops: bool = False,
            sample_all_records: bool = False,
            sample_start_idx: int | None = None,
            sample_end_idx: int | None = None,
            early_stop_iters: int = 3,
            use_final_op_quality: bool = False,
            seed: int = 42,
            exp_name: str | None = None,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # self.max_workers = self.get_parallel_max_workers()
        # TODO: undo
        # self.max_workers = 4
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.sample_all_ops = sample_all_ops
        self.sample_all_records = sample_all_records
        self.sample_start_idx = sample_start_idx
        self.sample_end_idx = sample_end_idx
        self.early_stop_iters = early_stop_iters
        self.use_final_op_quality = use_final_op_quality
        self.pick_output_fn = self.pick_champion_output
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name
    
    def compute_quality(
            self,
            record_set: DataRecordSet,
            expected_output: dict | None = None,
            champion_record_set: DataRecordSet | None = None,
            is_filter_op: bool = False,
            is_convert_op: bool = False,
        ) -> DataRecordSet:
        """
        Compute the quality for the given `record_set` by comparing it to the `expected_output`.

        Update the record_set by assigning the quality to each entry in its record_op_stats and
        returning the updated record_set.
        """
        # compute whether we can only use the champion
        only_using_champion = expected_output is None

        # if this operation is a failed convert
        if is_convert_op and len(record_set) == 0:
            record_set.record_op_stats[0].quality = 0.0

        # if this operation is a filter:
        # - we assign a quality of 1.0 if the record is in the expected outputs and it passes this filter
        # - we assign a quality of 0.0 if the record is in the expected outputs and it does NOT pass this filter
        # - we assign a quality relative to the champion / ensemble output if the record is not in the expected outputs
        # we cannot know for certain what the correct behavior is a given filter on a record which is not in the output
        # (unless it is the only filter in the plan), thus we only evaluate the filter based on its performance on
        # records which are in the output
        elif is_filter_op:
            # NOTE:
            # - we know that record_set.record_op_stats will contain a single entry for a filter op
            # - if we are using the champion, then champion_record_set will also contain a single entry for a filter op
            record_op_stats = record_set.record_op_stats[0]
            if only_using_champion:
                champion_record = champion_record_set[0]
                record_op_stats.quality = int(record_op_stats.passed_operator == champion_record.passed_operator)

            # - if we are using validation data, we may have multiple expected records in the expected_output for this source_idx,
            #   thus, if we can identify an exact match, we can use that to evaluate the filter's quality
            # - if we are using validation data but we *cannot* find an exact match, then we will once again use the champion record set
            else:
                # compute number of matches between this record's computed fields and this expected record's outputs
                found_match_in_output = False
                labels_dict_lst = expected_output["labels"] if isinstance(expected_output["labels"], list) else [expected_output["labels"]]
                for labels_dict in labels_dict_lst:
                    all_correct = True
                    for field, value in record_op_stats.record_state.items():
                        if value != labels_dict[field]:
                            all_correct = False
                            break

                    if all_correct:
                        found_match_in_output = True
                        break

                if found_match_in_output:
                    record_op_stats.quality = int(record_op_stats.passed_operator)
                else:
                    champion_record = champion_record_set[0]
                    record_op_stats.quality = int(record_op_stats.passed_operator == champion_record.passed_operator)

        # if this is a successful convert operation
        else:
            # NOTE: the following computation assumes we do not project out computed values
            #       (and that the validation examples provide all computed fields); even if
            #       a user program does add projection, we can ignore the projection on the
            #       validation dataset and use the champion model (as opposed to the validation
            #       output) for scoring fields which have their values projected out

            # create list of dictionaries of labels for each expected / champion output
            labels_dict_lst = []
            if only_using_champion:
                for champion_record in champion_record_set:
                    labels_dict_lst.append(champion_record.to_dict())
            else:
                labels_dict_lst = (
                    expected_output["labels"]
                    if isinstance(expected_output["labels"], list)
                    else [expected_output["labels"]]
                )

            # GREEDY ALGORITHM
            # for each record in the expected output, we look for the computed record which maximizes the quality metric;
            # once we've identified that computed record we remove it from consideration for the next expected output
            field_to_score_fn = {} if only_using_champion else expected_output["score_fn"]
            for labels_dict in labels_dict_lst:
                best_quality, best_record_op_stats = 0.0, None
                for record_op_stats in record_set.record_op_stats:
                    # if we already assigned this record a quality, skip it
                    if record_op_stats.quality is not None:
                        continue

                    # compute number of matches between this record's computed fields and this expected record's outputs
                    total_quality = 0
                    for field in record_op_stats.generated_fields:
                        computed_value = record_op_stats.record_state.get(field, None)
                        expected_value = labels_dict[field]

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


    def score_quality(
        self,
        op_set: list[PhysicalOperator],
        logical_op_id: str,
        execution_data: dict[str, dict[str, list[DataRecordSet]]],
        champion_outputs: dict[str, dict[str, DataRecordSet]],
        expected_outputs: dict[str, dict],
    ) -> list[RecordOpStats]:
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
        physical_op = op_set[0]
        is_filter_op = isinstance(physical_op, FilterOp)
        is_convert_op = isinstance(physical_op, ConvertOp)
        is_perfect_quality_op = (
            not isinstance(physical_op, LLMConvert)
            and not isinstance(physical_op, LLMFilter)
            and not isinstance(physical_op, RetrieveOp)
        )

        # pull out the execution data from this operator; place the upstream execution data in a new list
        this_op_execution_data = execution_data[logical_op_id]

        # compute quality of each output computed by this operator
        for source_idx, record_sets in this_op_execution_data.items():
            # NOTE
            # source_idx is a particular input, for which we may have computed multiple output record_sets;
            # each of these record_sets may contain more than one record (b/c one-to-many) and we have one
            # record_set per operator in the op_set

            # if this operation does not involve an LLM, every record_op_stats object gets perfect quality
            if is_perfect_quality_op:
                for record_set in record_sets:
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 1.0
                continue

            # get the expected output for this source_idx if we have one
            expected_output = (
                expected_outputs[source_idx]
                if expected_outputs is not None and source_idx in expected_outputs
                else None
            )

            # extract champion output for this record set
            champion_record_set = champion_outputs[logical_op_id][source_idx]

            # for each record_set produced by an operation, compute its quality
            for record_set in record_sets:
                record_set = self.compute_quality(record_set, expected_output, champion_record_set, is_filter_op, is_convert_op)

        # return the quality annotated record op stats
        return execution_data

    def pick_champion_output(self, op_set_record_sets: list[tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(op_set_record_sets) == 1:
            record_set, _ = op_set_record_sets[0]
            return record_set

        # find the operator with the highest average quality and return its record_set
        base_op_cost_est = OperatorCostEstimates(cardinality=1.0, cost_per_record=0.0, time_per_record=0.0, quality=1.0)
        champion_record_set, champion_quality = None, -1.0
        for record_set, op in op_set_record_sets:
            op_cost_estimates = op.naive_cost_estimates(base_op_cost_est)
            if op_cost_estimates.quality > champion_quality:
                champion_record_set, champion_quality = record_set, op_cost_estimates.quality

        return champion_record_set

    def pick_ensemble_output(self, op_set_record_sets: list[tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
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
                    idx_to_records[idx].append(record)

        # compute most common answer at each index
        out_records = []
        for idx in range(len(idx_to_records)):
            records = idx_to_records[idx]
            most_common_record = max(set(records), key=records.count)
            out_records.append(most_common_record)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, [])


    def pick_highest_quality_output(self, op_set_record_sets: list[tuple[DataRecordSet, PhysicalOperator]]) -> DataRecordSet:
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
            for idx in range(len(record_set)):
                record, record_op_stats = record_set[idx], record_set.record_op_stats[idx]
                if idx not in idx_to_records:
                    idx_to_records[idx] = [(record, record_op_stats)]
                else:
                    idx_to_records[idx].append((record, record_op_stats))

        # compute highest quality answer at each index
        out_records = []
        out_record_op_stats = []
        for idx in range(len(idx_to_records)):
            records_lst, record_op_stats_lst = zip(*idx_to_records[idx])
            max_quality_record, max_quality = records_lst[0], record_op_stats_lst[0].quality
            max_quality_stats = record_op_stats_lst[0]
            for record, record_op_stats in zip(records_lst[1:], record_op_stats_lst[1:]):
                record_quality = record_op_stats.quality
                if record_quality > max_quality:
                    max_quality_record = record
                    max_quality = record_quality
                    max_quality_stats = record_op_stats
            out_records.append(max_quality_record)
            out_record_op_stats.append(max_quality_stats)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, out_record_op_stats)

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan):
        """Execute a SentinelPlan according to strategy"""
        pass
