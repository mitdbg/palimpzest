import logging

import numpy as np

from palimpzest.core.data.dataclasses import SentinelPlanStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class OpSet:
    """
    This class represents the set of operators which are currently in the frontier for a given logical operator.
    Each operator in the frontier is an instance of a PhysicalOperator which either:

    1. lies on the Pareto frontier of the set of sampled operators, or
    2. has been sampled fewer than j times
    """

    def __init__(self, op_set: list[PhysicalOperator], source_indices: list[int]):
        # construct the set of operators
        self.ops = op_set

        # store the order in which we will sample the source records
        self.source_indices = source_indices

        # set the initial inputs for this logical operator
        is_scan_op = isinstance(op_set[0], ScanPhysicalOp)
        self.source_idx_to_input = {source_idx: [source_idx] for source_idx in self.source_indices} if is_scan_op else {}

    def get_op_input_pairs(self) -> list[PhysicalOperator, DataRecord | int | None]:
        """
        Returns the list of frontier operators and their next input to process. If there are
        any indices in `source_indices_to_sample` which this operator does not sample on its own, then
        we also have this frontier process that source_idx's input with its max quality operator.
        """
        # get the list of (op, source_idx) pairs which this operator needs to execute
        op_source_idx_pairs = []
        for op in self.ops:
            # construct list of inputs by looking up the input for the given source_idx
            for source_idx in self.source_indices:
                op_source_idx_pairs.append((op, source_idx))

        # fetch the corresponding (op, input) pairs
        op_input_pairs = []
        for op, source_idx in op_source_idx_pairs:
            op_input_pairs.extend([(op, input_record) for input_record in self.source_idx_to_input[source_idx]])

        return op_input_pairs

    def pick_highest_quality_output(self, record_sets: list[DataRecordSet]) -> DataRecordSet:
        # if there's only one operator in the set, we return its record_set
        if len(record_sets) == 1:
            return record_sets[0]

        # NOTE: I don't like that this assumes the models are consistent in
        #       how they order their record outputs for one-to-many converts;
        #       eventually we can try out more robust schemes to account for
        #       differences in ordering
        # aggregate records at each index in the response
        idx_to_records = {}
        for record_set in record_sets:
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

    def update_inputs(self, source_idx_to_record_sets: dict[int, DataRecordSet]):
        """
        Update the inputs for this logical operator based on the outputs of the previous logical operator.
        """
        for source_idx, record_sets in source_idx_to_record_sets.items():
            input = []
            max_quality_record_set = self.pick_highest_quality_output(record_sets)
            for record in max_quality_record_set:
                input.append(record if record.passed_operator else None)

            self.source_idx_to_input[source_idx] = input


class AllSamplingExecutionStrategy(SentinelExecutionStrategy):

    def _get_source_indices(self):
        """Get the list of source indices which the sentinel plan should execute over."""
        # create list of all source indices and shuffle it
        total_num_samples = len(self.val_datasource)
        source_indices = list(np.arange(total_num_samples))

        return source_indices

    def _execute_sentinel_plan(self,
            plan: SentinelPlan,
            op_sets: dict[str, OpSet],
            expected_outputs: dict[int, dict] | None,
            plan_stats: SentinelPlanStats,
        ) -> SentinelPlanStats:
        # execute operator sets in sequence
        for op_idx, (logical_op_id, op_set) in enumerate(plan):
            # get frontier ops and their next input
            op_input_pairs = op_sets[logical_op_id].get_op_input_pairs()

            # break out of the loop if op_input_pairs is empty, as this means all records have been filtered out
            if len(op_input_pairs) == 0:
                break

            # run sampled operators on sampled inputs
            source_idx_to_record_sets_and_ops, _ = self._execute_op_set(op_input_pairs)

            # FUTURE TODO: have this return the highest quality record set simply based on our posterior (or prior) belief on operator quality
            # get the target record set for each source_idx
            source_idx_to_target_record_set = self._get_target_record_sets(logical_op_id, source_idx_to_record_sets_and_ops, expected_outputs)

            # TODO: make consistent across here and RandomSampling
            # FUTURE TODO: move this outside of the loop (i.e. assume we only get quality label(s) after executing full program)
            # score the quality of each generated output
            physical_op_cls = op_set[0].__class__
            source_idx_to_record_sets = {
                source_idx: list(map(lambda tup: tup[0], record_sets_and_ops))
                for source_idx, record_sets_and_ops in source_idx_to_record_sets_and_ops.items()
            }
            source_idx_to_record_sets = self._score_quality(physical_op_cls, source_idx_to_record_sets, source_idx_to_target_record_set)

            # flatten the lists of records and record_op_stats
            all_records, all_record_op_stats = self._flatten_record_sets(source_idx_to_record_sets)

            # update plan stats
            plan_stats.add_record_op_stats(all_record_op_stats)

            # add records (which are not filtered) to the cache, if allowed
            self._add_records_to_cache(logical_op_id, all_records)

            # FUTURE TODO: simply set input based on source_idx_to_target_record_set (b/c we won't have scores computed)
            # provide the champion record sets as inputs to the next logical operator
            if op_idx + 1 < len(plan):
                next_logical_op_id = plan.logical_op_ids[op_idx + 1]
                op_sets[next_logical_op_id].update_inputs(source_idx_to_record_sets)

        # close the cache
        self._close_cache(plan.logical_op_ids)

        # finalize plan stats
        plan_stats.finish()

        return plan_stats

    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[int, dict] | None):
        """
        NOTE: this function currently requires us to set k and j properly in order to make
              comparison in our research against the corresponding sample budget in MAB.

        NOTE: the number of samples will slightly exceed the sample_budget if the number of operator
        calls does not perfectly match the sample_budget. This may cause some minor discrepancies with
        the progress manager as a result.
        """
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert all(isinstance(op, ScanPhysicalOp) for op in plan.operator_sets[0]), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # get list of source indices which can be sampled from
        source_indices = self._get_source_indices()

        # initialize set of physical operators for each logical operator
        op_sets = {
            logical_op_id: OpSet(op_set, source_indices)
            for logical_op_id, op_set in plan
        }

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, sample_budget=self.sample_budget, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _exeecute_sentinel_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail because
        #       the progress manager cannot get a handle to the console 
        try:
            # execute sentinel plan by sampling records and operators
            plan_stats = self._execute_sentinel_plan(plan, op_sets, expected_outputs, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing sentinel plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return plan_stats
