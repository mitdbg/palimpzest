from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import RecordOpStats, SentinelPlanStats
from palimpzest.core.elements.records import DataRecordSet
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.retrieve import RetrieveOp
from palimpzest.query.operators.scan import CacheScanDataOp, MarshalAndScanDataOp, ScanPhysicalOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.tools.logger import setup_logger
from palimpzest.utils.progress import create_progress_manager

logger = setup_logger(__name__)

class RandomSamplingExecutionStrategy(SentinelExecutionStrategy):

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

        # initialize progress manager
        self.progress_manager = create_progress_manager(plan, self.num_samples)

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
