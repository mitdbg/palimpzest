import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from palimpzest.constants import PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS
from palimpzest.core.data.dataclasses import RecordOpStats, SentinelPlanStats
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.filter import FilterOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.progress import create_progress_manager

logger = logging.getLogger(__name__)

class OpFrontier:
    """
    This class represents the set of operators which are currently in the frontier for a given logical operator.
    Each operator in the frontier is an instance of a PhysicalOperator which either:

    1. lies on the Pareto frontier of the set of sampled operators, or
    2. has been sampled fewer than j times
    """

    def __init__(self, op_set: list[PhysicalOperator], source_indices: list[int], k: int, j: int, seed: int, policy: Policy):
        # set k and j, which are the initial number of operators in the frontier and the
        # initial number of records to sample for each frontier operator
        self.k = min(k, len(op_set))
        self.j = min(j, len(source_indices))

        # store the policy that we are optimizing under
        self.policy = policy

        # get order in which we will sample physical operators for this logical operator
        sample_op_indices = self._get_op_index_order(op_set, seed)

        # construct the initial set of frontier and reservoir operators
        self.frontier_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[:k]]
        self.reservoir_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[k:]]

        # store the order in which we will sample the source records
        self.source_indices = source_indices

        # keep track of the number of times each operator has been sampled
        self.phys_op_id_to_num_samples = {op.get_op_id(): 0 for op in op_set}

        # keep track of the number of times the logical operator has been sampled
        self.total_num_samples = 0

        # set the initial inputs for this logical operator
        is_scan_op = isinstance(op_set[0], ScanPhysicalOp)
        self.source_idx_to_input = {source_idx: [source_idx] for source_idx in self.source_indices} if is_scan_op else {}

        # boolean indication of whether this is a logical filter
        self.is_filter_op = isinstance(op_set[0], FilterOp)

        # store all the execution data for this logical operator
        self.phys_op_id_to_record_op_stats_lst: dict[str, list[RecordOpStats]] = {}

    def _get_op_index_order(self, op_set: list[PhysicalOperator], seed: int) -> list[int]:
        """
        Returns a list of indices for the operators in the op_set.
        """
        rng = np.random.default_rng(seed=seed)
        op_indices = np.arange(len(op_set))
        rng.shuffle(op_indices)
        return op_indices

    def get_max_quality_op(self) -> PhysicalOperator:
        """
        Returns the operator in the frontier with the highest (estimated) quality.
        """
        max_quality_op, max_avg_quality = None, None
        for op in self.frontier_ops:
            total_quality = sum([record_op_stats.quality for record_op_stats in self.phys_op_id_to_record_op_stats_lst[op.get_op_id()]])
            avg_op_quality = total_quality / len(self.phys_op_id_to_num_samples[op.get_op_id()])
            if max_avg_quality is None or avg_op_quality > max_avg_quality:
                max_quality_op = op
                max_avg_quality = avg_op_quality

        return max_quality_op

    def _get_op_source_idx_pairs(self) -> list[tuple[PhysicalOperator, int]]:
        """
        Returns a list of tuples for (op, source_idx) which this operator needs to execute
        in the next iteration.
        """
        op_source_idx_pairs = []
        for op in self.frontier_ops:
            # execute new operators on first j source indices, and previously sampled operators on one additional source_idx
            current_num_samples = self.phys_op_id_to_num_samples[op.get_op_id()]
            num_new_samples = 1 if current_num_samples > 0 else self.j
            num_new_samples = min(num_new_samples, len(self.source_indices) - current_num_samples)

            # construct list of inputs by looking up the input for the given source_idx
            for sample_idx in range(current_num_samples, current_num_samples + num_new_samples):
                source_idx = self.source_indices[sample_idx]
                op_source_idx_pairs.append((op, source_idx))
        
        return op_source_idx_pairs

    def get_source_indices_for_next_iteration(self) -> set[int]:
        """
        Returns the set of source indices which need to be sampled for the next iteration.
        """
        op_source_idx_pairs = self._get_op_source_idx_pairs()
        return set(map(lambda tup: tup[1], op_source_idx_pairs))

    def get_frontier_op_input_pairs(self, source_indices_to_sample: set[int]) -> list[PhysicalOperator, DataRecord | int | None]:
        """
        Returns the list of frontier operators and their next input to process. If there are
        any indices in `source_indices_to_sample` which this operator does not sample on its own, then
        we also have this frontier process that source_idx's input with its max quality operator.
        """
        # get the list of (op, source_idx) pairs which this operator needs to execute
        op_source_idx_pairs = self._get_op_source_idx_pairs()

        # if there are any source_idxs in source_indices_to_sample where are not sampled
        # by this operator, apply the max quality operator to them
        sampled_source_indices = set(map(lambda tup: tup[1], op_source_idx_pairs))
        unsampled_source_indices = source_indices_to_sample - sampled_source_indices
        max_quality_op = self.get_max_quality_op()
        for source_idx in unsampled_source_indices:
            op_source_idx_pairs.append((max_quality_op, source_idx))

        # fetch the corresponding (op, input) pairs
        op_input_pairs = []
        for op, source_idx in op_source_idx_pairs:
            op_input_pairs.extend([(op, input_record) for input_record in self.source_idx_to_input[source_idx]])
            self.phys_op_id_to_num_samples[op.get_op_id()] += len(op_input_pairs)
            self.total_num_samples += len(op_input_pairs)

        return op_input_pairs

    def update_frontier(self, source_idx_to_record_sets: dict[int, list[DataRecordSet]]) -> None:
        """
        Update the set of frontier operators, pulling in new ones from the reservoir as needed.
        This function will:
        1. Compute the mean, LCB, and UCB for the cost, time, quality, and selectivity of each frontier operator
        2. Compute the pareto optimal set of frontier operators (using the mean values)
        3. Update the frontier and reservoir sets of operators based on their LCB/UCB overlap with the pareto frontier
        """
        # compute metrics for each physical operator in source_idx_to_record_sets
        op_metrics = {}

        # compute selectivity for each physical operator
        phys_op_to_num_inputs, phys_op_to_num_outputs = {}, {}
        for _, record_sets in source_idx_to_record_sets.items():
            for record_set in record_sets:
                op_id = record_set.record_op_stats[0].op_id
                num_outputs = sum([record_op_stats.passed_operator for record_op_stats in record_set.record_op_stats])
                if op_id not in phys_op_to_num_inputs:
                    phys_op_to_num_inputs[op_id] = 1
                    phys_op_to_num_outputs[op_id] = num_outputs
                else:
                    phys_op_to_num_inputs[op_id] += 1
                    phys_op_to_num_outputs[op_id] += num_outputs

        phys_op_to_mean_selectivity = {
            op_id: phys_op_to_num_outputs[op_id] / phys_op_to_num_inputs[op_id]
            for op_id in phys_op_to_num_inputs
        }

        # compute average cost, time, and quality
        phys_op_to_costs, phys_op_to_times, phys_op_to_qualities = {}, {}, {}
        for _, record_sets in source_idx_to_record_sets.items():
            for record_set in record_sets:
                for record_op_stats in record_set.record_op_stats:
                    op_id = record_op_stats.op_id
                    cost = record_op_stats.cost_per_record
                    time = record_op_stats.time_per_record
                    quality = record_op_stats.quality
                    if op_id not in phys_op_to_costs:
                        phys_op_to_costs[op_id] = [cost]
                        phys_op_to_times[op_id] = [time]
                        phys_op_to_qualities[op_id] = [quality]
                    else:
                        phys_op_to_costs[op_id].append(cost)
                        phys_op_to_times[op_id].append(time)
                        phys_op_to_qualities[op_id].append(quality)

        phys_op_to_mean_cost = {op: np.mean(costs) for op, costs in phys_op_to_costs.items()}
        phys_op_to_mean_time = {op: np.mean(times) for op, times in phys_op_to_times.items()}
        phys_op_to_mean_quality = {op: np.mean(qualities) for op, qualities in phys_op_to_qualities.items()}

        # compute average, LCB, and UCB of each operator; the confidence bounds depend upon
        # the computation of the alpha parameter, which we scale to be 0.5 * the mean (of means)
        # of the metric across all operators in this operator set
        cost_alpha = 0.5 * np.mean([mean_cost for mean_cost in phys_op_to_mean_cost.values()])
        time_alpha = 0.5 * np.mean([mean_time for mean_time in phys_op_to_mean_time.values()])
        quality_alpha = 0.5 * np.mean([mean_quality for mean_quality in phys_op_to_mean_quality.values()])
        selectivity_alpha = 0.5 * np.mean([mean_selectivity for mean_selectivity in phys_op_to_mean_selectivity.values()])

        op_metrics = {}
        for op_id in phys_op_to_costs:
            sample_ratio = np.sqrt(np.log(self.total_num_samples) / self.phys_op_id_to_num_samples[op_id])
            exploration_terms = np.array([cost_alpha * sample_ratio, time_alpha * sample_ratio, quality_alpha * sample_ratio, selectivity_alpha * sample_ratio])
            mean_terms = (phys_op_to_mean_cost[op_id], phys_op_to_mean_time[op_id], phys_op_to_mean_quality[op_id], phys_op_to_mean_selectivity[op_id])

            # NOTE: we could clip these; however I will not do so for now to allow for arbitrary quality metric(s)
            lcb_terms = mean_terms - exploration_terms
            ucb_terms = mean_terms + exploration_terms
            op_metrics[op_id] = {"mean": mean_terms, "lcb": lcb_terms, "ucb": ucb_terms}

        # get the tuple representation of this policy
        policy_dict = self.policy.get_dict()

        # compute the pareto optimal set of operators
        pareto_op_set = set()
        for op_id, metrics in op_metrics.items():
            cost, time, quality, selectivity = metrics["mean"]
            pareto_frontier = True

            # check if any other operator dominates op_id
            for other_op_id, other_metrics in op_metrics.items():
                other_cost, other_time, other_quality, other_selectivity = other_metrics["mean"]
                if op_id == other_op_id:
                    continue

                # if op_id is dominated by other_op_id, set pareto_frontier = False and break
                # NOTE: here we use a strict inequality (instead of the usual <= or >=) because
                #       all ops which have equal cost / time / quality / sel. should not be
                #       filtered out from sampling by our logic in this function
                cost_dominated = True if policy_dict["cost"] == 0.0 else other_cost < cost
                time_dominated = True if policy_dict["time"] == 0.0 else other_time < time
                quality_dominated = True if policy_dict["quality"] == 0.0 else other_quality > quality
                selectivity_dominated = True if not self.is_filter_op else other_selectivity < selectivity
                if cost_dominated and time_dominated and quality_dominated and selectivity_dominated:
                    pareto_frontier = False
                    break

            # add op_id to pareto frontier if it's not dominated
            if pareto_frontier:
                pareto_op_set.add(op_id)

        # iterate over frontier ops and replace any which do not overlap with pareto frontier
        new_frontier_ops, new_reservoir_ops = [], []
        num_dropped_from_frontier = 0
        for op in self.frontier_ops:
            op_id = op.get_op_id()

            # if this op is fully sampled, remove it from the frontier
            if self.phys_op_id_to_num_samples[op_id] == len(self.source_indices):
                num_dropped_from_frontier += 1
                continue

            # if this op is pareto optimal keep it in our frontier ops
            if op_id in pareto_op_set:
                new_frontier_ops.append(op)
                continue

            # otherwise, if this op overlaps with an op on the pareto frontier, keep it in our frontier ops
            # NOTE: for now, we perform an optimistic comparison with the ucb/lcb
            pareto_frontier = True
            op_cost = op_metrics[op_id]["lcb"][0]
            op_time = op_metrics[op_id]["lcb"][1]
            op_quality = op_metrics[op_id]["ucb"][2]
            op_selectivity = op_metrics[op_id]["lcb"][3]
            for pareto_op_id in pareto_op_set:
                pareto_cost = op_metrics[pareto_op_id]["ucb"][0]
                pareto_time = op_metrics[pareto_op_id]["ucb"][1]
                pareto_quality = op_metrics[pareto_op_id]["lcb"][2]
                pareto_selectivity = op_metrics[pareto_op_id]["ucb"][3]

                # if op_id is dominated by pareto_op_id, set pareto_frontier = False and break
                cost_dominated = True if policy_dict["cost"] == 0.0 else pareto_cost <= op_cost
                time_dominated = True if policy_dict["time"] == 0.0 else pareto_time <= op_time
                quality_dominated = True if policy_dict["quality"] == 0.0 else pareto_quality >= op_quality
                selectivity_dominated = True if not self.is_filter_op else pareto_selectivity <= op_selectivity
                if cost_dominated and time_dominated and quality_dominated and selectivity_dominated:
                    pareto_frontier = False
                    break
            
            # add op_id to pareto frontier if it's not dominated
            if pareto_frontier:
                new_frontier_ops.append(op)
            else:
                num_dropped_from_frontier += 1

        # replace the ops dropped from the frontier with new ops from the reservoir
        num_dropped_from_frontier = min(num_dropped_from_frontier, len(self.reservoir_ops))
        for _ in range(num_dropped_from_frontier):
            new_op = self.reservoir_ops.pop(0)
            new_frontier_ops.append(new_op)

        # update the frontier and reservoir ops
        self.frontier_ops = new_frontier_ops
        self.reservoir_ops = new_reservoir_ops

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
        input = []
        for source_idx, record_sets in source_idx_to_record_sets.items():
            max_quality_record_set = self.pick_highest_quality_output(record_sets)
            for record in max_quality_record_set:
                if record.passed_operator:
                    input.append(record)

            self.source_idx_to_input[source_idx] = input


# TODO: post-submission we will need to modify this to:
# - submit all inputs for aggregate operators
# - handle limits
class MABExecutionStrategy(SentinelExecutionStrategy):
    """
    This class implements the Multi-Armed Bandit (MAB) execution strategy for SentinelQueryProcessors.
    """

    def _execute_op_set(self, op_input_pairs: list[PhysicalOperator, DataRecord | int]) -> dict[int, list[tuple[DataRecordSet, PhysicalOperator]]]:
        def execute_op_wrapper(operator, input):
            record_set = operator(input)
            return record_set, operator, input

        # TODO: modify unit tests to always have record_op_stats so we can use record_op_stats for source_idx
        # for scan operators, `input` will be the source_idx
        def get_source_idx(input):
            return input.source_idx if isinstance(input, DataRecord) else input

        # initialize mapping from source indices to output record sets
        source_idx_to_record_sets_and_ops = {get_source_idx(input): [] for _, input in op_input_pairs}

        # create thread pool w/max workers and run futures over worker pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # create futures
            futures = [
                executor.submit(execute_op_wrapper, operator, input)
                for operator, input in op_input_pairs
            ]
            output_record_sets = [future.result() for future in futures]

            # compute mapping from source_idx to record sets for all operators and for champion operator
            for record_set, operator, input in output_record_sets:
                # get the source_idx associated with this input record;
                source_idx = get_source_idx(input)

                # add record_set to mapping from source_idx --> record_sets
                source_idx_to_record_sets_and_ops[source_idx].append((record_set, operator))

        return source_idx_to_record_sets_and_ops

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

    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[int, dict] | None):
        """
        """
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert all(isinstance(op, ScanPhysicalOp) for op in plan.operator_sets[0]), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # TODO: add support for multiple plans and sentinel plans
        # initialize progress manager
        # self.progress_manager = create_progress_manager(plan, self.num_samples)

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # shuffle the indices of records to sample
        total_num_samples = len(self.val_datasource)
        shuffled_source_indices = [int(idx) for idx in np.arange(total_num_samples)]
        self.rng.shuffle(shuffled_source_indices)

        # initialize frontier for each logical operator
        op_frontiers = {
            logical_op_id: OpFrontier(op_set, shuffled_source_indices, self.k, self.j, self.seed, self.policy)
            for logical_op_id, op_set in plan
        }

        # sample records and operators and update the frontiers
        samples_drawn = 0
        while samples_drawn < self.sample_budget:
            # pre-compute the set of source indices which will need to be sampled
            source_indices_to_sample = set()
            for op_frontier in op_frontiers.values():
                source_indices = op_frontier.get_source_indices_for_next_iteration()
                source_indices_to_sample.update(source_indices)

            # execute operator sets in sequence
            for op_idx, (logical_op_id, op_set) in enumerate(plan):
                # get frontier ops and their next input
                frontier_op_input_pairs = op_frontiers[logical_op_id].get_frontier_op_input_pairs(source_indices_to_sample)

                # break out of the loop if frontier_op_input_pairs is empty, as this means all records have been filtered out
                if len(frontier_op_input_pairs) == 0:
                    break

                # run sampled operators on sampled inputs
                source_idx_to_record_sets_and_ops = self._execute_op_set(frontier_op_input_pairs)

                # TODO: keep a cache of outputs for (source_idx, logical_op_id) and return best one (or expected output)
                # FUTURE TODO: change this logic to simply select an input for the next operator
                # get the champion record set for each source_idx
                source_idx_to_champion_record_set = self._get_champion_record_sets(source_idx_to_record_sets_and_ops)

                # TODO: make consistent across here and RandomSampling
                # FUTURE TODO: move this outside of the loop (i.e. assume we only get quality label(s) after executing full program)
                # score the quality of each generated output
                physical_op_cls = op_set[0].__class__
                source_idx_to_record_sets = {
                    source_idx: list(map(lambda tup: tup[0], record_sets_and_ops))
                    for source_idx, record_sets_and_ops in source_idx_to_record_sets_and_ops.items()
                }
                source_idx_to_record_sets = self._score_quality(physical_op_cls, source_idx_to_record_sets, source_idx_to_champion_record_set, expected_outputs)

                # flatten the lists of records and record_op_stats
                all_records, all_record_op_stats = self._flatten_record_sets(source_idx_to_record_sets)

                # update plan stats
                plan_stats.add_record_op_stats(all_record_op_stats)

                # add records (which are not filtered) to the cache, if allowed
                self._add_records_to_cache(logical_op_id, all_records)

                # provide the champion record sets as inputs to the next logical operator
                if op_idx + 1 < len(plan):
                    next_logical_op_id = plan.logical_op_ids[op_idx + 1]
                    op_frontiers[next_logical_op_id].update_inputs(source_idx_to_record_sets)
                
                # update the (pareto) frontier for each set of operators
                op_frontiers[logical_op_id].update_frontier(source_idx_to_record_sets)

            # FUTURE TODO: apply scoring outside of for loop

            # update the number of samples drawn to be the max across all logical operators
            samples_drawn = max([op_frontier.total_num_samples for op_frontier in op_frontiers.values()])

        # close the cache
        self._close_cache(plan.logical_op_ids)

        # finalize plan stats
        plan_stats.finish()

        return plan_stats
