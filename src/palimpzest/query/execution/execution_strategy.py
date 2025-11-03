import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.constants import Cardinality
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, PlanStats, SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import PZSentinelProgressManager
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)

class BaseExecutionStrategy:
    def __init__(self,
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 batch_size: int | None = None,
                 num_samples: int | None = None,
                 verbose: bool = False,
                 progress: bool = True,
                 *args,
                 **kwargs):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.batch_size = batch_size
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
        policy: Policy,
        k: int = 6,
        j: int = 4,
        sample_budget: int = 100,
        sample_cost_budget: float | None = None,
        priors: dict | None = None,
        use_final_op_quality: bool = False,
        seed: int = 42,
        exp_name: str | None = None,
        dont_use_priors: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.j = j
        self.sample_budget = sample_budget
        self.sample_cost_budget = sample_cost_budget
        self.policy = policy
        self.priors = priors
        self.use_final_op_quality = use_final_op_quality
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name
        self.dont_use_priors = dont_use_priors

        # general cache which maps hash(logical_op_id, phys_op_id, hash(input)) --> record_set
        self.cache: dict[int, DataRecordSet] = {}

        # progress manager used to track progress of the execution
        self.progress_manager: PZSentinelProgressManager | None = None

    def _score_quality(
        self,
        validator: Validator,
        source_indices_to_record_sets: dict[tuple[str], list[tuple[DataRecordSet, PhysicalOperator]]],
    ) -> tuple[dict[int, list[DataRecordSet]], GenerationStats]:
        # extract information about the logical operation performed at this stage of the sentinel plan;
        # NOTE: we can infer these fields from context clues, but in the long-term we should have a more
        #       principled way of getting these directly from attributes either stored in the sentinel_plan
        #       or in the PhysicalOperator
        def is_perfect_quality_op(op: PhysicalOperator):
            return (
                not isinstance(op, LLMConvert)
                and not isinstance(op, LLMFilter)
                and not isinstance(op, TopKOp)
                and not isinstance(op, JoinOp)
            )

        # create minimal set of futures necessary to compute quality of each output record
        futures, full_hashes, full_hash_to_bool_output = [], set(), {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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

                    # create future for map
                    if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0].to_dict(project_cols=fields)
                        output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                        full_hash = f"{hash(input_record)}{hash(output_str)}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_map, op, fields, input_record, output, full_hash))

                    # create future for flat map
                    elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output, output_strs = [], []
                        for data_record in record_set.data_records:
                            output.append(data_record.to_dict(project_cols=fields))
                            output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                        full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_flat_map, op, fields, input_record, output, full_hash))

                    # create future for top-k
                    elif isinstance(op, TopKOp):
                        fields = op.generated_fields
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0].to_dict(project_cols=fields)
                        output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                        full_hash = f"{hash(input_record)}{hash(output_str)}"
                        if full_hash not in full_hashes:
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_topk, op, fields, input_record, output, full_hash))

                    # create future for filter
                    elif isinstance(op, LLMFilter):
                        filter_str = op.filter_obj.filter_condition
                        input_record: DataRecord = record_set.input
                        output = record_set.data_records[0]._passed_operator
                        full_hash = f"{filter_str}{hash(input_record)}"
                        if full_hash not in full_hashes:
                            full_hash_to_bool_output[full_hash] = output
                            full_hashes.add(full_hash)
                            futures.append(executor.submit(validator._score_filter, op, filter_str, input_record, output, full_hash))

                    # create future for join
                    elif isinstance(op, JoinOp):
                        condition = op.condition
                        for left_idx, left_input_record in enumerate(record_set.input[0]):
                            for right_idx, right_input_record in enumerate(record_set.input[1]):
                                record_idx = left_idx * len(record_set.input[1]) + right_idx
                                output = record_set.data_records[record_idx]._passed_operator
                                full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                                if full_hash not in full_hashes:
                                    full_hash_to_bool_output[full_hash] = output
                                    full_hashes.add(full_hash)
                                    futures.append(executor.submit(validator._score_join, op, condition, left_input_record, right_input_record, output, full_hash))

        # collect results from futures
        full_hash_to_score, validation_gen_stats = {}, GenerationStats()
        for future in as_completed(futures):
            score, gen_stats, full_hash = future.result()
            full_hash_to_score[full_hash] = score
            validation_gen_stats += gen_stats

        # compute quality of each output computed by this operator
        for _, record_set_tuples in source_indices_to_record_sets.items():
            for record_set, op in record_set_tuples:
                if is_perfect_quality_op(op) or len(record_set) == 0:
                    continue

                if isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_ONE:
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]

                elif isinstance(op, LLMConvert) and op.cardinality is Cardinality.ONE_TO_MANY:
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_strs = []
                    for data_record in record_set.data_records:
                        output_strs.append(data_record.to_json_str(project_cols=fields, bytes_to_str=True, sorted=True))
                    full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                    score = full_hash_to_score[full_hash]
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = score

                # TODO: this scoring function will (likely) bias towards small values of k since it
                # measures precision and not recall / F1; will need to revisit this in the future
                elif isinstance(op, TopKOp):
                    fields = op.generated_fields
                    input_record: DataRecord = record_set.input
                    output_str = record_set.data_records[0].to_json_str(project_cols=fields, bytes_to_str=True, sorted=True)
                    full_hash = f"{hash(input_record)}{hash(output_str)}"
                    score = full_hash_to_score[full_hash]
                    record_set.record_op_stats[0].quality = score

                elif isinstance(op, LLMFilter):
                    filter_str = op.filter_obj.filter_condition
                    input_record: DataRecord = record_set.input
                    output = record_set.data_records[0]._passed_operator
                    full_hash = f"{filter_str}{hash(input_record)}"
                    if output == full_hash_to_bool_output[full_hash]:
                        record_set.record_op_stats[0].quality = full_hash_to_score[full_hash]
                    else:
                        record_set.record_op_stats[0].quality = 1.0 - full_hash_to_score[full_hash]

                elif isinstance(op, JoinOp):
                    condition = op.condition
                    for left_idx, left_input_record in enumerate(record_set.input[0]):
                        for right_idx, right_input_record in enumerate(record_set.input[1]):
                            record_idx = left_idx * len(record_set.input[1]) + right_idx
                            output = record_set.data_records[record_idx]._passed_operator
                            full_hash = f"{condition}{hash(left_input_record)}{hash(right_input_record)}"
                            if output == full_hash_to_bool_output[full_hash]:
                                record_set.record_op_stats[record_idx].quality = full_hash_to_score[full_hash]
                            else:
                                record_set.record_op_stats[record_idx].quality = 1.0 - full_hash_to_score[full_hash]

        # return the quality annotated record sets
        return source_indices_to_record_sets, validation_gen_stats

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
            for future in as_completed(futures):
                # update output record sets
                record_set, operator, source_indices, input = future.result()

                # if the operator is a join, get record_set from tuple output
                if isinstance(operator, JoinOp):
                    record_set = record_set[0]

                output_record_sets.append((record_set, operator, source_indices, input))

                # update cache
                op_input_hash = get_hash(operator, input)
                self.cache[op_input_hash] = (record_set, operator)

                # update progress manager
                if self._is_llm_op(operator):
                    num_llm_ops += 1
                    self.progress_manager.incr(unique_logical_op_id, num_samples=1, total_cost=record_set.get_total_cost())

            # update mapping from source_indices to record sets and operators
            for record_set, operator, source_indices, input in output_record_sets:
                # add record_set to mapping from source_indices --> record_sets
                record_set.input = input
                source_indices_to_record_sets_and_ops[source_indices].append((record_set, operator, True))

        return source_indices_to_record_sets_and_ops, num_llm_ops

    def _is_llm_op(self, physical_op: PhysicalOperator) -> bool:
        is_llm_convert = isinstance(physical_op, LLMConvert)
        is_llm_filter = isinstance(physical_op, LLMFilter)
        is_llm_topk = isinstance(physical_op, TopKOp) and isinstance(physical_op.index, Collection)
        is_llm_join = isinstance(physical_op, JoinOp)
        return is_llm_convert or is_llm_filter or is_llm_topk or is_llm_join

    @abstractmethod
    def execute_sentinel_plan(self, sentinel_plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator) -> SentinelPlanStats:
        """Execute a SentinelPlan according to strategy"""
        pass
