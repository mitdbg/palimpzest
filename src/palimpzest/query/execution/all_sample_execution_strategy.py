import logging

import numpy as np

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import SentinelPlanStats
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.progress import create_progress_manager
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)

class OpSet:
    """
    This class represents the set of operators which are currently in the frontier for a given logical operator.
    Each operator in the frontier is an instance of a PhysicalOperator which either:

    1. lies on the Pareto frontier of the set of sampled operators, or
    2. has been sampled fewer than j times
    """

    def __init__(self, op_set: list[PhysicalOperator], source_unique_logical_op_ids: list[str], source_indices: list[int]):
        # construct the set of operators
        self.ops = op_set

        # store the order in which we will sample the source records
        self.source_indices = source_indices

        # boolean indication of the type of operator in this OpSet
        sample_op = op_set[0]
        self.is_scan_op = isinstance(sample_op, (ScanPhysicalOp, ContextScanOp))
        self.is_filter_op = isinstance(sample_op, FilterOp)
        self.is_aggregate_op = isinstance(sample_op, AggregateOp)
        self.is_llm_join = isinstance(sample_op, JoinOp)

        # set the initial inputs for this logical operator
        self.source_indices_to_inputs = {source_unique_logical_op_id: {} for source_unique_logical_op_id in source_unique_logical_op_ids}
        if self.is_scan_op:
            self.source_indices_to_inputs["source"] = {source_idx: [int(source_idx.split("-")[-1])] for source_idx in self.source_indices}

    def get_op_inputs(self) -> list[PhysicalOperator, DataRecord | int | None]:
        """
        Returns the list of frontier operators and their next input to process.
        """
        # if this is an aggregate, run on every input
        if self.is_aggregate_op:
            op = self.ops[0]
            all_inputs = []
            for _, source_indices_to_inputs in self.source_indices_to_inputs.items():
                for _, inputs in source_indices_to_inputs.items():
                    all_inputs.extend(inputs)
            return [(op, tuple(), all_inputs)]

        # if this is an un-optimized (non-scan, non-join) operator, flatten inputs and run on each one
        elif not self.is_scan_op and not self.is_llm_join and len(self.ops) == 1:
            op_inputs = []
            op = self.ops[0]
            for _, source_indices_to_inputs in self.source_indices_to_inputs.items():
                for source_indices, inputs in source_indices_to_inputs.items():
                    for input in inputs:
                        op_inputs.append((op, source_indices, input))
            return op_inputs

        # get the list of (op, source_indices) pairs which this operator needs to execute
        op_source_indices_pairs = []
        for op in self.ops:
            # construct list of inputs by looking up the input for the given source_indices
            for source_indices in self.source_indices:
                op_source_indices_pairs.append((op, source_indices))

        # construct the op inputs
        op_inputs = []
        if self.is_llm_join:
            left_source_unique_logical_op_id, right_source_unique_logical_op_id = list(self.source_indices_to_inputs)
            left_source_indices_to_inputs = self.source_indices_to_inputs[left_source_unique_logical_op_id]
            right_source_indices_to_inputs = self.source_indices_to_inputs[right_source_unique_logical_op_id]
            for op, source_indices in op_source_indices_pairs:
                left_source_indices = source_indices[0]
                right_source_indices = source_indices[1]
                left_inputs = left_source_indices_to_inputs.get(left_source_indices, [])
                right_inputs = right_source_indices_to_inputs.get(right_source_indices, [])
                if len(left_inputs) > 0 and len(right_inputs) > 0:
                    op_inputs.append((op, (left_source_indices, right_source_indices), (left_inputs, right_inputs)))
            return op_inputs

        # if operator is not a join
        source_unique_logical_op_id = list(self.source_indices_to_inputs)[0]
        op_inputs = [
            (op, source_indices, input)
            for op, source_indices in op_source_indices_pairs
            for input in self.source_indices_to_inputs[source_unique_logical_op_id].get(source_indices, [])
        ]

        return op_inputs

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
                input.append(record if record._passed_operator else None)

            self.source_indices_to_inputs[source_idx] = input

class AllSamplingExecutionStrategy(SentinelExecutionStrategy):

    def _execute_sentinel_plan(self,
            plan: SentinelPlan,
            op_sets: dict[str, OpSet],
            validator: Validator,
            plan_stats: SentinelPlanStats,
        ) -> SentinelPlanStats:
        # execute operator sets in sequence
        for topo_idx, (logical_op_id, _) in enumerate(plan):
            # compute unique logical op id within plan
            unique_logical_op_id = f"{topo_idx}-{logical_op_id}"

            # get frontier ops and their next input
            op_inputs = op_sets[logical_op_id].get_op_inputs()

            # break out of the loop if op_inputs is empty, as this means all records have been filtered out
            if len(op_inputs) == 0:
                break

            # run sampled operators on sampled inputs
            source_indices_to_record_set_tuples, _ = self._execute_op_set(unique_logical_op_id, op_inputs)

            # score the quality of each generated output
            source_indices_to_all_record_sets = {
                    source_indices: [(record_set, op) for record_set, op, _ in record_set_tuples]
                    for source_indices, record_set_tuples in source_indices_to_record_set_tuples.items()
                }
            source_indices_to_all_record_sets, val_gen_stats = self._score_quality(validator, source_indices_to_all_record_sets)

            # remove records that were read from the execution cache before adding to record op stats
            new_record_op_stats = []
            for _, record_set_tuples in source_indices_to_record_set_tuples.items():
                for record_set, _, is_new in record_set_tuples:
                    if is_new:
                        new_record_op_stats.extend(record_set.record_op_stats)

            # update plan stats
            plan_stats.add_record_op_stats(unique_logical_op_id, new_record_op_stats)
            plan_stats.add_validation_gen_stats(unique_logical_op_id, val_gen_stats)

            # provide the best record sets as inputs to the next logical operator
            next_unique_logical_op_id = plan.get_next_unique_logical_op_id(unique_logical_op_id)
            if next_unique_logical_op_id is not None:
                source_indices_to_all_record_sets = {
                    source_indices: [record_set for record_set, _ in record_set_tuples]
                    for source_indices, record_set_tuples in source_indices_to_all_record_sets.items()
                }
                op_sets[next_unique_logical_op_id].update_inputs(unique_logical_op_id, source_indices_to_all_record_sets)

        # finalize plan stats
        plan_stats.finish()

        return plan_stats

    def execute_sentinel_plan(self, plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator): # expected_outputs: dict[int, dict] | None):
        """
        NOTE: this function currently requires us to set k and j properly in order to make
              comparison in our research against the corresponding sample budget in MAB.

        NOTE: the number of samples will slightly exceed the sample_budget if the number of operator
        calls does not perfectly match the sample_budget. This may cause some minor discrepancies with
        the progress manager as a result.
        """
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # get lists of source indices
        dataset_id_to_source_indices = {}
        for dataset_id, dataset in train_dataset.items():
            total_num_samples = len(dataset)
            source_indices = [f"{dataset_id}---{int(idx)}" for idx in np.arange(total_num_samples)]
            dataset_id_to_source_indices[dataset_id] = source_indices

        # initialize set of physical operators for each logical operator
        op_sets = {}
        for topo_idx, (logical_op_id, op_set) in enumerate(plan):
            unique_logical_op_id = f"{topo_idx}-{logical_op_id}"
            source_unique_logical_op_ids = plan.get_source_unique_logical_op_ids(unique_logical_op_id)
            sample_op = op_set[0]
            if isinstance(sample_op, (ScanPhysicalOp, ContextScanOp)):
                root_dataset_ids = plan.get_root_dataset_ids(unique_logical_op_id)
                assert len(root_dataset_ids) == 1, f"Scan for {sample_op} has {len(root_dataset_ids)} > 1 root dataset ids"
                root_dataset_id = root_dataset_ids[0]
                source_indices = dataset_id_to_source_indices[root_dataset_id]
                op_sets[unique_logical_op_id] = OpSet(op_set, source_unique_logical_op_ids, source_indices)
            elif isinstance(sample_op, JoinOp):
                assert len(source_unique_logical_op_ids) == 2, f"Join for {sample_op} has {len(source_unique_logical_op_ids)} != 2 source logical operators"
                left_source_indices = op_sets[source_unique_logical_op_ids[0]].source_indices
                right_source_indices = op_sets[source_unique_logical_op_ids[1]].source_indices
                source_indices = []
                for left_source_idx in left_source_indices:
                    for right_source_idx in right_source_indices:
                        source_indices.append((left_source_idx, right_source_idx))
                op_sets[unique_logical_op_id] = OpSet(op_set, source_unique_logical_op_ids, source_indices)
            else:
                source_indices = op_sets[source_unique_logical_op_ids[0]].source_indices
                op_sets[unique_logical_op_id] = OpSet(op_set, source_unique_logical_op_ids, source_indices)

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, sample_budget=self.sample_budget, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_sentinel_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail because
        #       the progress manager cannot get a handle to the console 
        try:
            # execute sentinel plan by sampling records and operators
            plan_stats = self._execute_sentinel_plan(plan, op_sets, validator, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing sentinel plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return plan_stats
