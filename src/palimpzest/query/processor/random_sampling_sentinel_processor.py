import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import (
    ExecutionStats,
    OperatorCostEstimates,
    RecordOpStats,
    SentinelPlanStats,
)
from palimpzest.core.elements.records import DataRecordCollection, DataRecordSet
from palimpzest.policy import Policy
from palimpzest.query.execution.parallel_execution_strategy import ParallelExecutionStrategy
from palimpzest.query.execution.single_threaded_execution_strategy import (
    PipelinedSingleThreadExecutionStrategy,
    SequentialSingleThreadExecutionStrategy,
)
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import CacheScanDataOp, MarshalAndScanDataOp, ScanPhysicalOp
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
from palimpzest.query.optimizer.optimizer_strategy import OptimizationStrategyType
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.sets import Set

logger = logging.getLogger(__name__)

class RandomSamplingSentinelQueryProcessor(QueryProcessor):
    """
    """
    def __init__(
            self,
            k: int,
            sample_budget: int,
            sample_all_ops: bool = False,
            sample_all_records: bool = False,
            sample_start_idx: int | None = None,
            sample_end_idx: int | None = None,
            use_final_op_quality: bool = False,
            seed: int = 42,
            exp_name: str | None = None,
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # self.max_workers = self.get_parallel_max_workers()
        # TODO: undo
        # self.max_workers = 1
        self.k = k
        self.sample_budget = sample_budget
        self.j = int(sample_budget / k)
        self.sample_all_ops = sample_all_ops
        self.sample_all_records = sample_all_records
        self.sample_start_idx = sample_start_idx
        self.sample_end_idx = sample_end_idx
        self.use_final_op_quality = use_final_op_quality
        self.pick_output_fn = self.pick_ensemble_output
        self.rng = np.random.default_rng(seed=seed)
        self.exp_name = exp_name
        logger.info(f"Initialized RandomSamplingSentinelQueryProcessor with config: {self.config}")


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
            operator_sets: list[list[PhysicalOperator]],
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
        op_set = operator_sets[-1]
        physical_op = op_set[0]
        is_source_op = isinstance(physical_op, (MarshalAndScanDataOp, CacheScanDataOp))
        is_filter_op = isinstance(physical_op, FilterOp)
        is_convert_op = isinstance(physical_op, ConvertOp)
        is_perfect_quality_op = (
            not isinstance(physical_op, LLMConvert)
            and not isinstance(physical_op, LLMFilter)
            and not isinstance(physical_op, RetrieveOp)
        )
        logical_op_id = physical_op.logical_op_id

        # if this logical_op_id is not in the execution_data (because all upstream records were filtered), return
        if logical_op_id not in execution_data:
            return execution_data

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

        # if this operator is a source op (i.e. has no input logical operator), return the execution data
        if is_source_op:
            return execution_data

        # recursively call the function on the next logical operator until you reach a scan
        execution_data = self.score_quality(operator_sets[:-1], execution_data, champion_outputs, expected_outputs)

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


    def execute_op_set(self, candidates, op_set):
        def execute_op_wrapper(operator, candidate):
            record_set = operator(candidate)
            return record_set, operator, candidate

        # TODO: post-submission we will need to modify this to:
        # - submit all candidates for aggregate operators
        # - handle limits
        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = []
            for candidate in candidates:
                for operator in op_set:
                    future = executor.submit(execute_op_wrapper, operator, candidate)
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
                    # get the result and add it to the output records set
                    record_set, operator, candidate = future.result()
                    output_record_sets.append((record_set, operator, candidate))

                # update list of futures
                futures = not_done_futures

            # compute mapping from source_idx to record sets for all operators and for champion operator
            all_record_sets, champion_record_sets = {}, {}
            for candidate in candidates:
                candidate_output_record_sets = []
                for record_set, operator, candidate_ in output_record_sets:
                    if candidate == candidate_:
                        candidate_output_record_sets.append((record_set, operator))

                # select the champion (i.e. best) record_set from all the record sets computed for this operator
                champion_record_set = self.pick_output_fn(candidate_output_record_sets)

                # get the source_idx associated with this input record
                source_idx = candidate.source_idx

                # add champion record_set to mapping from source_idx --> champion record_set
                champion_record_sets[source_idx] = champion_record_set

                # add all record_sets computed for this source_idx to mapping from source_idx --> record_sets
                all_record_sets[source_idx] = [tup[0] for tup in candidate_output_record_sets]

        return all_record_sets, champion_record_sets


    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[str, dict], policy: Policy):
        """
        """
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert all(isinstance(op, ScanPhysicalOp) for op in plan.operator_sets[0]), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # # initialize progress manager
        # self.progress_manager = create_progress_manager(plan, self.num_samples)

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # sample validation records
        total_num_samples = len(self.val_datasource)
        source_indices = np.arange(total_num_samples)
        if self.sample_start_idx is not None:
            assert self.sample_end_idx is not None, "Specified `sample_start_idx` without specifying `sample_end_idx`"
            source_indices = source_indices[self.sample_start_idx:self.sample_end_idx]
        elif not self.sample_all_records:
            self.rng.shuffle(source_indices)
            j = min(self.j, len(source_indices))
            source_indices = source_indices[:j]

        # initialize output variables
        all_outputs, champion_outputs = {}, {}

        # create initial set of candidates for source scan operator
        candidates = []
        for source_idx in source_indices:
            candidates.append(source_idx)

        # NOTE: because we need to dynamically create sample matrices for each operator,
        #       sentinel execution must be executed one operator at a time (i.e. sequentially)
        # execute operator sets in sequence
        for op_idx, (logical_op_id, op_set) in enumerate(plan):
            next_logical_op_id = plan.logical_op_ids[op_idx + 1] if op_idx + 1 < len(plan) else None

            # sample k optimizations
            k = min(self.k, len(op_set)) if not self.sample_all_ops else len(op_set)
            sampled_ops = self.rng.choice(op_set, size=k, replace=False)

            # run sampled operators on sampled candidates
            source_idx_to_record_sets, source_idx_to_champion_record_set = self.execute_op_set(candidates, sampled_ops)

            # update all_outputs and champion_outputs dictionary
            if logical_op_id not in all_outputs:
                all_outputs[logical_op_id] = source_idx_to_record_sets
                champion_outputs[logical_op_id] = source_idx_to_champion_record_set
            else:
                for source_idx, record_sets in source_idx_to_record_sets.items():
                    if source_idx not in all_outputs[logical_op_id]:
                        all_outputs[logical_op_id][source_idx] = record_sets
                        champion_outputs[logical_op_id][source_idx] = source_idx_to_champion_record_set[source_idx]
                    else:
                        all_outputs[logical_op_id][source_idx].extend(record_sets)
                        champion_outputs[logical_op_id][source_idx].extend(source_idx_to_champion_record_set[source_idx])

            # flatten lists of records and record_op_stats
            all_records, all_record_op_stats = [], []
            for _, record_sets in source_idx_to_record_sets.items():
                for record_set in record_sets:
                    all_records.extend(record_set.data_records)
                    all_record_op_stats.extend(record_set.record_op_stats)

            # update plan stats
            plan_stats.add_record_op_stats(all_record_op_stats)

            # add records (which are not filtered) to the cache, if allowed
            if self.cache:
                for record in all_records:
                    if getattr(record, "passed_operator", True):
                        # self.datadir.append_cache(logical_op_id, record)
                        pass

            # update candidates for next operator; we use champion outputs as input
            candidates = []
            if next_logical_op_id is not None:
                for _, record_set in source_idx_to_champion_record_set.items():
                    for record in record_set:
                        if isinstance(op_set[0], FilterOp) and not record.passed_operator:
                            continue
                        candidates.append(record)

            # if we've filtered out all records, terminate early
            if next_logical_op_id is not None and candidates == []:
                break

        # compute quality for each operator
        all_outputs = self.score_quality(plan.operator_sets, all_outputs, champion_outputs, expected_outputs)

        # if caching was allowed, close the cache
        if self.cache:
            for _, _ in plan:
                # self.datadir.close_cache(logical_op_id)
                pass

        # finalize plan stats
        plan_stats.finish()

        return all_outputs, plan_stats


    def generate_sample_observations(self, sentinel_plan: SentinelPlan, policy: Policy):
        """
        This function is responsible for generating sample observation data which can be
        consumed by the CostModel.

        To accomplish this, we construct a special sentinel plan using the Optimizer which is
        capable of executing any valid physical implementation of a Filter or Convert operator
        on each record.
        """
        # if we're using validation data, get the set of expected output records
        expected_outputs = {}
        for source_idx in range(len(self.val_datasource)):
            # TODO: make sure execute_op_set uses self.val_datasource
            expected_output = self.val_datasource[source_idx]
            expected_outputs[source_idx] = expected_output

        # run sentinel plan
        execution_data, plan_stats = self.execute_sentinel_plan(sentinel_plan, expected_outputs, policy)

        return execution_data, plan_stats

    
    def create_sentinel_plan(self, dataset: Set, policy: Policy) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # TODO: explicitly pull up filters; for SIGMOD we can explicitly write plans w/filters pulled up

        # create a new optimizer and update its strategy to SENTINEL
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_strategy(OptimizationStrategyType.SENTINEL)

        # create copy of dataset, but change its data source to the validation data source
        dataset = dataset.copy()
        dataset._set_data_source(self.val_datasource)

        # get the sentinel plan for the given dataset
        sentinel_plans = optimizer.optimize(dataset, policy)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan


    def execute(self) -> DataRecordCollection:
        logger.info("Executing RandomSamplingSentinelQueryProcessor")
        execution_start_time = time.time()

        # for now, enforce that we are using validation data; we can relax this after paper submission
        if self.val_datasource is None:
            raise Exception("Make sure you are using validation data with MABSentinelExecutionEngine")

        # if cache is False, make sure we do not re-use codegen examples
        if not self.cache:
            # self.clear_cached_examples()
            pass

        # create sentinel plan
        sentinel_plan = self.create_sentinel_plan(self.dataset, self.policy)

        # generate sample execution data
        all_execution_data, plan_stats = self.generate_sample_observations(sentinel_plan, self.policy)

        # put sentinel plan execution stats into list and prepare list of output records
        all_plan_stats = [plan_stats]
        all_records = []

        # construct the CostModel with any sample execution data we've gathered
        cost_model = SampleBasedCostModel(sentinel_plan, all_execution_data, self.verbose, self.exp_name)
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_cost_model(cost_model)
        total_optimization_time = time.time() - execution_start_time

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, self.policy, optimizer)
        all_records.extend(records)
        all_plan_stats.extend(plan_stats)

        # aggregate plan stats
        aggregate_plan_stats = self.aggregate_plan_stats(all_plan_stats)

        # add sentinel records and plan stats (if captured) to plan execution data
        execution_stats = ExecutionStats(
            execution_id=self.execution_id(),
            plan_stats=aggregate_plan_stats,
            total_optimization_time=total_optimization_time,
            total_execution_time=time.time() - execution_start_time,
            total_execution_cost=sum(list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))),
            plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
        )

        result =  DataRecordCollection(all_records, execution_stats=execution_stats)
        logger.info("Done executing RandomSamplingSentinelQueryProcessor")
        logger.debug(f"Result: {result}")
        return result


class RandomSamplingSentinelSequentialSingleThreadProcessor(RandomSamplingSentinelQueryProcessor, SequentialSingleThreadExecutionStrategy):
    """
    This class performs sentinel execution while executing plans in a sequential, single-threaded fashion.
    """
    def __init__(self, *args, **kwargs):
        RandomSamplingSentinelQueryProcessor.__init__(self, *args, **kwargs)
        SequentialSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            verbose=self.verbose
        )
        logger.info("Created RandomSamplingSentinelSequentialSingleThreadProcessor")

class RandomSamplingSentinelPipelinedSingleThreadProcessor(RandomSamplingSentinelQueryProcessor, PipelinedSingleThreadExecutionStrategy):
    """
    This class performs sentinel execution while executing plans in a pipelined fashion.
    """
    def __init__(self, *args, **kwargs):
        RandomSamplingSentinelQueryProcessor.__init__(self, *args, **kwargs)
        PipelinedSingleThreadExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            verbose=self.verbose
        )
        logger.info("Created RandomSamplingSentinelPipelinedSingleThreadProcessor")

class RandomSamplingSentinelParallelProcessor(RandomSamplingSentinelQueryProcessor, ParallelExecutionStrategy):
    """
    This class performs sentinel execution while executing plans in a parallel fashion.
    """
    def __init__(self, *args, **kwargs):
        RandomSamplingSentinelQueryProcessor.__init__(self, *args, **kwargs)
        ParallelExecutionStrategy.__init__(
            self,
            scan_start_idx=self.scan_start_idx,
            max_workers=self.max_workers,
            verbose=self.verbose
        )
        logger.info("Created RandomSamplingSentinelParallelProcessor")
