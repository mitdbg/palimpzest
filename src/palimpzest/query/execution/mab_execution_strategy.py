import logging

import numpy as np

from palimpzest.core.data.dataclasses import OperatorStats, SentinelPlanStats
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

    def __init__(self, op_set: list[PhysicalOperator], source_indices: list[int], k: int, j: int, seed: int, policy: Policy, priors: dict | None = None):
        # set k and j, which are the initial number of operators in the frontier and the
        # initial number of records to sample for each frontier operator
        self.k = min(k, len(op_set))
        self.j = min(j, len(source_indices))

        # store the policy that we are optimizing under
        self.policy = policy

        # store the prior beliefs on operator performance (if provided)
        self.priors = priors

        # get order in which we will sample physical operators for this logical operator
        sample_op_indices = self._get_op_index_order(op_set, seed)

        # construct the initial set of frontier and reservoir operators
        self.frontier_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[:self.k]]
        self.reservoir_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[self.k:]]
        self.off_frontier_ops: list[PhysicalOperator] = []

        # store the order in which we will sample the source records
        self.source_indices = source_indices

        # keep track of the source ids processed by each physical operator
        self.full_op_id_to_sources_processed = {op.get_full_op_id(): set() for op in op_set}

        # set the initial inputs for this logical operator
        is_scan_op = isinstance(op_set[0], ScanPhysicalOp)
        self.source_idx_to_input = {source_idx: [source_idx] for source_idx in self.source_indices} if is_scan_op else {}

        # boolean indication of whether this is a logical filter
        self.is_filter_op = isinstance(op_set[0], FilterOp)

    def get_frontier_ops(self) -> list[PhysicalOperator]:
        """
        Returns the set of frontier operators for this OpFrontier.
        """
        return self.frontier_ops

    def _compute_op_id_to_pareto_distance(self, priors: dict[str, dict[str, float]]) -> dict[str, float]:
        """
        Return l2-distance for each operator from the pareto frontier.
        """
        # get the dictionary representation of this poicy
        policy_dict = self.policy.get_dict()

        # compute the pareto optimal set of operators
        pareto_op_set = set()
        for op_id, metrics in priors.items():
            cost, time, quality = metrics["cost"], metrics["time"], metrics["quality"]
            pareto_frontier = True

            # check if any other operator dominates op_id
            for other_op_id, other_metrics in priors.items():
                other_cost, other_time, other_quality = other_metrics["cost"], other_metrics["time"], other_metrics["quality"]
                if op_id == other_op_id:
                    continue

                # if op_id is dominated by other_op_id, set pareto_frontier = False and break
                # NOTE: here we use a strict inequality (instead of the usual <= or >=) because
                #       all ops which have equal cost / time / quality / sel. should not be
                #       filtered out from sampling by our logic in this function
                cost_dominated = True if policy_dict["cost"] == 0.0 else other_cost < cost
                time_dominated = True if policy_dict["time"] == 0.0 else other_time < time
                quality_dominated = True if policy_dict["quality"] == 0.0 else other_quality > quality
                if cost_dominated and time_dominated and quality_dominated:
                    pareto_frontier = False
                    break

            # add op_id to pareto frontier if it's not dominated
            if pareto_frontier:
                pareto_op_set.add(op_id)

        # compute the shortest distance from each operator to the pareto frontier
        op_id_to_pareto_distance = {}
        for op_id, metrics in priors.items():
            # set distance to 0.0 if this operator is on the pareto frontier
            if op_id in pareto_op_set:
                op_id_to_pareto_distance[op_id] = 0.0
                continue

            # otherwise, compute min_dist to pareto operators
            min_dist = None
            cost, time, quality = metrics["cost"], metrics["time"], metrics["quality"]
            for pareto_op_id in pareto_op_set:
                pareto_cost, pareto_time, pareto_quality = priors[pareto_op_id]["cost"], priors[pareto_op_id]["time"], priors[pareto_op_id]["quality"]

                cost_dist_squared = 0.0 if policy_dict["cost"] == 0.0 else (cost - pareto_cost) ** 2
                time_dist_squared = 0.0 if policy_dict["time"] == 0.0 else (time - pareto_time) ** 2
                quality_dist_squared = 0.0 if policy_dict["quality"] == 0.0 else (quality - pareto_quality) ** 2
                dist = np.sqrt(cost_dist_squared + time_dist_squared + quality_dist_squared)
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            # set minimum distance for this operator
            op_id_to_pareto_distance[op_id] = min_dist
        
        return op_id_to_pareto_distance

    def _get_op_index_order(self, op_set: list[PhysicalOperator], seed: int) -> list[int]:
        """
        Returns a list of indices for the operators in the op_set.
        """
        if self.priors is None or any([op_id not in self.priors for op_id in map(lambda op: op.get_op_id(), op_set)]):
            rng = np.random.default_rng(seed=seed)
            op_indices = np.arange(len(op_set))
            rng.shuffle(op_indices)
            return op_indices

        # NOTE: self.priors is a dictionary with format:
        # {op_id: {"quality": quality, "cost": cost, "time": time}}

        # compute mean and std. dev. for each field
        qualities = [op_priors["quality"] for op_priors in self.priors.values()]
        costs = [op_priors["cost"] for op_priors in self.priors.values()]
        times = [op_priors["time"] for op_priors in self.priors.values()]
        metric_to_mean = {"quality": np.mean(qualities), "cost": np.mean(costs), "time": np.mean(times)}
        metric_to_std = {"quality": np.std(qualities), "cost": np.std(costs), "time": np.std(times)}

        # normalize the scale of each field to be the same
        for _, op_priors in self.priors.items():
            for metric, value in op_priors.items():
                if metric_to_std[metric] == 0.0:
                    op_priors[metric] = metric_to_mean[metric]
                else:
                    op_priors[metric] = (value - metric_to_mean[metric]) / metric_to_std[metric]

        # then, we compute the l2-distance from the pareto frontier for each operator
        op_id_to_distance = self._compute_op_id_to_pareto_distance(self.priors)

        # compute tuple for every operator, invert quality so ascending sort puts
        # best operator first: (op_id, dist, -1 * quality, cost, time);
        op_tuples = []
        for op in op_set:
            op_id = op.get_op_id()
            op_priors = self.priors[op_id]
            op_tuple = (op_id, op_id_to_distance[op_id], -1 * op_priors["quality"], op_priors["cost"], op_priors["time"])
            op_tuples.append(op_tuple)

        # sort tuples on distance, then second dim
        second_dim_idx = None
        if self.policy.get_primary_metric() == "quality":
            second_dim_idx = 2
        elif self.policy.get_primary_metric() == "cost":
            second_dim_idx = 3
        elif self.policy.get_primary_metric() == "time":
            second_dim_idx = 4

        # sort based on distance from pareto frontier; break ties with performance on max / min metric
        op_tuples = sorted(op_tuples, key=lambda x: (x[1], x[second_dim_idx]))

        # return final list of op indices in sample order
        op_id_to_idx = {op.get_op_id(): idx for idx, op in enumerate(op_set)}
        op_indices = [op_id_to_idx[op_tuple[0]] for op_tuple in op_tuples]

        return op_indices

    def _get_op_source_idx_pairs(self) -> list[tuple[PhysicalOperator, int]]:
        """
        Returns a list of tuples for (op, source_idx) which this operator needs to execute
        in the next iteration.
        """
        op_source_idx_pairs = []
        for op in self.frontier_ops:
            # execute new operators on first j source indices, and previously sampled operators on one additional source_idx
            num_processed = len(self.full_op_id_to_sources_processed[op.get_full_op_id()])
            num_new_samples = 1 if num_processed > 0 else self.j
            num_new_samples = min(num_new_samples, len(self.source_indices) - num_processed)
            assert num_new_samples >= 0, "Number of new samples must be non-negative"

            # construct list of inputs by looking up the input for the given source_idx
            samples_added = 0
            for source_idx in self.source_indices:
                if source_idx in self.full_op_id_to_sources_processed[op.get_full_op_id()]:
                    continue

                if samples_added == num_new_samples:
                    break

                # construct the (op, source_idx) for this source_idx
                op_source_idx_pairs.append((op, source_idx))
                samples_added += 1

        return op_source_idx_pairs

    def get_source_indices_for_next_iteration(self) -> set[int]:
        """
        Returns the set of source indices which need to be sampled for the next iteration.
        """
        op_source_idx_pairs = self._get_op_source_idx_pairs()
        return set(map(lambda tup: tup[1], op_source_idx_pairs))

    def get_frontier_op_input_pairs(self, source_indices_to_sample: set[int], max_quality_op: PhysicalOperator) -> list[PhysicalOperator, DataRecord | int | None]:
        """
        Returns the list of frontier operators and their next input to process. If there are
        any indices in `source_indices_to_sample` which this operator does not sample on its own, then
        we also have this frontier process that source_idx's input with its max quality operator.
        """
        # get the list of (op, source_idx) pairs which this operator needs to execute
        op_source_idx_pairs = self._get_op_source_idx_pairs()

        # if there are any source_idxs in source_indices_to_sample which are not sampled
        # by this operator, apply the max quality operator (and any other frontier operators
        # with no samples)
        sampled_source_indices = set(map(lambda tup: tup[1], op_source_idx_pairs))
        unsampled_source_indices = source_indices_to_sample - sampled_source_indices
        for source_idx in unsampled_source_indices:
            op_source_idx_pairs.append((max_quality_op, source_idx))
            for op in self.frontier_ops:
                if len(self.full_op_id_to_sources_processed[op.get_full_op_id()]) == 0 and op.get_full_op_id() != max_quality_op.get_full_op_id():
                    op_source_idx_pairs.append((op, source_idx))

        # fetch the corresponding (op, input) pairs
        op_input_pairs = [
            (op, input)
            for op, source_idx in op_source_idx_pairs
            for input in self.source_idx_to_input[source_idx]
        ]

        return op_input_pairs

    def update_frontier(self, logical_op_id: str, plan_stats: SentinelPlanStats) -> None:
        """
        Update the set of frontier operators, pulling in new ones from the reservoir as needed.
        This function will:
        1. Compute the mean, LCB, and UCB for the cost, time, quality, and selectivity of each frontier operator
        2. Compute the pareto optimal set of frontier operators (using the mean values)
        3. Update the frontier and reservoir sets of operators based on their LCB/UCB overlap with the pareto frontier
        """
        # NOTE: downstream operators may end up re-computing the same record_id with a diff. input as upstream
        #       upstream operators change; in this case, we de-duplicate record_op_stats with identical record_ids
        #       and keep the one with the maximum quality
        # get a mapping from full_op_id --> list[RecordOpStats]
        full_op_id_to_op_stats: dict[str, OperatorStats] = plan_stats.operator_stats.get(logical_op_id, {})
        full_op_id_to_record_op_stats = {}
        for full_op_id, op_stats in full_op_id_to_op_stats.items():
            # skip over operators which have not been sampled
            if len(op_stats.record_op_stats_lst) == 0:
                continue

            # compute mapping from record_id to highest quality record op stats
            record_id_to_max_quality_record_op_stats = {}
            for record_op_stats in op_stats.record_op_stats_lst:
                record_id = record_op_stats.record_id
                if record_id not in record_id_to_max_quality_record_op_stats:  # noqa: SIM114
                    record_id_to_max_quality_record_op_stats[record_id] = record_op_stats

                elif record_op_stats.quality > record_id_to_max_quality_record_op_stats[record_id].quality:
                    record_id_to_max_quality_record_op_stats[record_id] = record_op_stats

            # compute final list of record op stats
            full_op_id_to_record_op_stats[full_op_id] = list(record_id_to_max_quality_record_op_stats.values())

        # compute mapping of physical op to num samples and total samples drawn;
        # also update the set of source indices which have been processed by each physical operator
        full_op_id_to_num_samples, total_num_samples = {}, 0
        for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items():
            # update the set of source indices processed
            for record_op_stats in record_op_stats_lst:
                self.full_op_id_to_sources_processed[full_op_id].add(record_op_stats.record_source_idx)

            # compute the number of samples as the number of source indices processed
            num_samples = len(self.full_op_id_to_sources_processed[full_op_id])
            full_op_id_to_num_samples[full_op_id] = num_samples
            total_num_samples += num_samples

        # compute avg. selectivity, cost, time, and quality for each physical operator
        def total_output(record_op_stats_lst):
            return sum([record_op_stats.passed_operator for record_op_stats in record_op_stats_lst])

        def total_input(record_op_stats_lst):
            return len(set([record_op_stats.record_parent_id for record_op_stats in record_op_stats_lst]))

        full_op_id_to_mean_selectivity = {
            full_op_id: total_output(record_op_stats_lst) / total_input(record_op_stats_lst)
            for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items()
        }
        full_op_id_to_mean_cost = {
            full_op_id: np.mean([record_op_stats.cost_per_record for record_op_stats in record_op_stats_lst])
            for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items()
        }
        full_op_id_to_mean_time = {
            full_op_id: np.mean([record_op_stats.time_per_record for record_op_stats in record_op_stats_lst])
            for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items()
        }
        full_op_id_to_mean_quality = {
            full_op_id: np.mean([record_op_stats.quality for record_op_stats in record_op_stats_lst])
            for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items()
        }

        # # compute average, LCB, and UCB of each operator; the confidence bounds depend upon
        # # the computation of the alpha parameter, which we scale to be 0.5 * the mean (of means)
        # # of the metric across all operators in this operator set
        # cost_alpha = 0.5 * np.mean([mean_cost for mean_cost in full_op_id_to_mean_cost.values()])
        # time_alpha = 0.5 * np.mean([mean_time for mean_time in full_op_id_to_mean_time.values()])
        # quality_alpha = 0.5 * np.mean([mean_quality for mean_quality in full_op_id_to_mean_quality.values()])
        # selectivity_alpha = 0.5 * np.mean([mean_selectivity for mean_selectivity in full_op_id_to_mean_selectivity.values()])
        cost_alpha = 0.5 * (np.max(list(full_op_id_to_mean_cost.values())) - np.min(list(full_op_id_to_mean_cost.values())))
        time_alpha = 0.5 * (np.max(list(full_op_id_to_mean_time.values())) - np.min(list(full_op_id_to_mean_time.values())))
        quality_alpha = 0.5 * (np.max(list(full_op_id_to_mean_quality.values())) - np.min(list(full_op_id_to_mean_quality.values())))
        selectivity_alpha = 0.5 * (np.max(list(full_op_id_to_mean_selectivity.values())) - np.min(list(full_op_id_to_mean_selectivity.values())))

        # compute metrics for each physical operator
        op_metrics = {}
        for full_op_id in full_op_id_to_record_op_stats:
            sample_ratio = np.sqrt(np.log(total_num_samples) / full_op_id_to_num_samples[full_op_id])
            exploration_terms = np.array([cost_alpha * sample_ratio, time_alpha * sample_ratio, quality_alpha * sample_ratio, selectivity_alpha * sample_ratio])
            mean_terms = (full_op_id_to_mean_cost[full_op_id], full_op_id_to_mean_time[full_op_id], full_op_id_to_mean_quality[full_op_id], full_op_id_to_mean_selectivity[full_op_id])

            # NOTE: we could clip these; however I will not do so for now to allow for arbitrary quality metric(s)
            lcb_terms = mean_terms - exploration_terms
            ucb_terms = mean_terms + exploration_terms
            op_metrics[full_op_id] = {"mean": mean_terms, "lcb": lcb_terms, "ucb": ucb_terms}

        # get the tuple representation of this policy
        policy_dict = self.policy.get_dict()

        # compute the pareto optimal set of operators
        pareto_op_set = set()
        for full_op_id, metrics in op_metrics.items():
            cost, time, quality, selectivity = metrics["mean"]
            pareto_frontier = True

            # check if any other operator dominates full_op_id
            for other_full_op_id, other_metrics in op_metrics.items():
                other_cost, other_time, other_quality, other_selectivity = other_metrics["mean"]
                if full_op_id == other_full_op_id:
                    continue

                # if full_op_id is dominated by other_full_op_id, set pareto_frontier = False and break
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

            # add full_op_id to pareto frontier if it's not dominated
            if pareto_frontier:
                pareto_op_set.add(full_op_id)

        # iterate over op metrics and compute the new frontier set of operators
        new_frontier_full_op_ids = set()
        for full_op_id, metrics in op_metrics.items():

            # if this op is fully sampled, do not keep it on the frontier
            if full_op_id_to_num_samples[full_op_id] == len(self.source_indices):
                continue

            # if this op is pareto optimal keep it in our frontier ops
            if full_op_id in pareto_op_set:
                new_frontier_full_op_ids.add(full_op_id)
                continue

            # otherwise, if this op overlaps with an op on the pareto frontier, keep it in our frontier ops
            # NOTE: for now, we perform an optimistic comparison with the ucb/lcb
            pareto_frontier = True
            op_cost, op_time, _, op_selectivity = metrics["lcb"]
            op_quality = metrics["ucb"][2]
            for pareto_full_op_id in pareto_op_set:
                pareto_cost, pareto_time, _, pareto_selectivity = op_metrics[pareto_full_op_id]["ucb"]
                pareto_quality = op_metrics[pareto_full_op_id]["lcb"][2]

                # if full_op_id is dominated by pareto_full_op_id, set pareto_frontier = False and break
                cost_dominated = True if policy_dict["cost"] == 0.0 else pareto_cost <= op_cost
                time_dominated = True if policy_dict["time"] == 0.0 else pareto_time <= op_time
                quality_dominated = True if policy_dict["quality"] == 0.0 else pareto_quality >= op_quality
                selectivity_dominated = True if not self.is_filter_op else pareto_selectivity <= op_selectivity
                if cost_dominated and time_dominated and quality_dominated and selectivity_dominated:
                    pareto_frontier = False
                    break

            # add full_op_id to pareto frontier if it's not dominated
            if pareto_frontier:
                new_frontier_full_op_ids.add(full_op_id)

        # for operators that were in the frontier, keep them in the frontier if they
        # are still pareto optimal, otherwise, move them to the end of the reservoir
        new_frontier_ops = []
        for op in self.frontier_ops:
            if op.get_full_op_id() in new_frontier_full_op_ids:
                new_frontier_ops.append(op)
            else:
                self.off_frontier_ops.append(op)

        # if there are operators we previously sampled which are now back on the frontier
        # add them to the frontier, otherwise, put them back in the off_frontier_ops
        new_off_frontier_ops = []
        for op in self.off_frontier_ops:
            if op.get_full_op_id() in new_frontier_full_op_ids:
                new_frontier_ops.append(op)
            else:
                new_off_frontier_ops.append(op)

        # finally, if we have fewer than k operators in the frontier, sample new operators
        # from the reservoir and put them in the frontier
        while len(new_frontier_ops) < self.k and len(self.reservoir_ops) > 0:
            new_op = self.reservoir_ops.pop(0)
            new_frontier_ops.append(new_op)

        # update the frontier and off frontier ops
        self.frontier_ops = new_frontier_ops
        self.off_frontier_ops = new_off_frontier_ops

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


# TODO: post-submission we will need to modify this to:
# - submit all inputs for aggregate operators
# - handle limits
class MABExecutionStrategy(SentinelExecutionStrategy):
    """
    This class implements the Multi-Armed Bandit (MAB) execution strategy for SentinelQueryProcessors.

    NOTE: the number of samples will slightly exceed the sample_budget if the number of operator
    calls does not perfectly match the sample_budget. This may cause some minor discrepancies with
    the progress manager as a result.
    """
    def _get_max_quality_op(self, logical_op_id: str, op_frontiers: dict[str, OpFrontier], plan_stats: SentinelPlanStats) -> PhysicalOperator:
        """
        Returns the operator in the frontier with the highest (estimated) quality.
        """
        # get the operators in the frontier set for this logical_op_id
        frontier_ops = op_frontiers[logical_op_id].get_frontier_ops()

        # get a mapping from full_op_id --> list[RecordOpStats]
        full_op_id_to_op_stats: dict[str, OperatorStats] = plan_stats.operator_stats.get(logical_op_id, {})
        full_op_id_to_record_op_stats = {
            full_op_id: op_stats.record_op_stats_lst
            for full_op_id, op_stats in full_op_id_to_op_stats.items()
        }

        # iterate over the frontier ops and return the one with the highest quality
        max_quality_op, max_avg_quality = None, None
        for op in frontier_ops:
            op_quality_stats = []
            if op.get_full_op_id() in full_op_id_to_record_op_stats:
                op_quality_stats = [record_op_stats.quality for record_op_stats in full_op_id_to_record_op_stats[op.get_full_op_id()]]
            avg_op_quality = sum(op_quality_stats) / len(op_quality_stats) if len(op_quality_stats) > 0 else 0.0
            if max_avg_quality is None or avg_op_quality > max_avg_quality:
                max_quality_op = op
                max_avg_quality = avg_op_quality

        return max_quality_op

    def _execute_sentinel_plan(
            self,
            plan: SentinelPlan,
            op_frontiers: dict[str, OpFrontier],
            expected_outputs: dict[int, dict] | None,
            plan_stats: SentinelPlanStats,
        ) -> SentinelPlanStats:
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
                # use the execution cache to determine the maximum quality operator for this logical_op_id
                max_quality_op = self._get_max_quality_op(logical_op_id, op_frontiers, plan_stats)

                # TODO: can have None as an operator if _get_max_quality_op returns None
                # get frontier ops and their next input
                frontier_op_input_pairs = op_frontiers[logical_op_id].get_frontier_op_input_pairs(source_indices_to_sample, max_quality_op)
                frontier_op_input_pairs = list(filter(lambda tup: tup[1] is not None, frontier_op_input_pairs))

                # break out of the loop if frontier_op_input_pairs is empty, as this means all records have been filtered out
                if len(frontier_op_input_pairs) == 0:
                    break

                # run sampled operators on sampled inputs and update the number of samples drawn
                source_idx_to_record_set_tuples, num_llm_ops = self._execute_op_set(frontier_op_input_pairs)
                samples_drawn += num_llm_ops

                # FUTURE TODO: have this return the highest quality record set simply based on our posterior (or prior) belief on operator quality
                # get the target record set for each source_idx
                source_idx_to_target_record_set = self._get_target_record_sets(logical_op_id, source_idx_to_record_set_tuples, expected_outputs)

                # FUTURE TODO: move this outside of the loop (i.e. assume we only get quality label(s) after executing full program)
                # score the quality of each generated output
                physical_op_cls = op_set[0].__class__
                source_idx_to_all_record_sets = {
                    source_idx: [record_set for record_set, _, _ in record_set_tuples]
                    for source_idx, record_set_tuples in source_idx_to_record_set_tuples.items()
                }
                source_idx_to_all_record_sets = self._score_quality(physical_op_cls, source_idx_to_all_record_sets, source_idx_to_target_record_set)

                # flatten the lists of newly computed records and record_op_stats
                source_idx_to_new_record_sets = {
                    source_idx: [record_set for record_set, _, is_new in record_set_tuples if is_new]
                    for source_idx, record_set_tuples in source_idx_to_record_set_tuples.items()
                }
                new_records, new_record_op_stats = self._flatten_record_sets(source_idx_to_new_record_sets)

                # update the number of samples drawn for each operator

                # update plan stats
                plan_stats.add_record_op_stats(new_record_op_stats)

                # add records (which are not filtered) to the cache, if allowed
                self._add_records_to_cache(logical_op_id, new_records)

                # FUTURE TODO: simply set input based on source_idx_to_target_record_set (b/c we won't have scores computed)
                # provide the champion record sets as inputs to the next logical operator
                if op_idx + 1 < len(plan):
                    next_logical_op_id = plan.logical_op_ids[op_idx + 1]
                    op_frontiers[next_logical_op_id].update_inputs(source_idx_to_all_record_sets)

                # update the (pareto) frontier for each set of operators
                op_frontiers[logical_op_id].update_frontier(logical_op_id, plan_stats)

            # FUTURE TODO: score op quality based on final outputs

        # close the cache
        self._close_cache(plan.logical_op_ids)

        # finalize plan stats
        plan_stats.finish()

        return plan_stats


    def execute_sentinel_plan(self, plan: SentinelPlan, expected_outputs: dict[int, dict] | None):
        # for now, assert that the first operator in the plan is a ScanPhysicalOp
        assert all(isinstance(op, ScanPhysicalOp) for op in plan.operator_sets[0]), "First operator in physical plan must be a ScanPhysicalOp"
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")
        logger.info(f"Plan Details: {plan}")

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # shuffle the indices of records to sample
        total_num_samples = len(self.val_datasource)
        shuffled_source_indices = [int(idx) for idx in np.arange(total_num_samples)]
        self.rng.shuffle(shuffled_source_indices)

        # initialize frontier for each logical operator
        op_frontiers = {
            logical_op_id: OpFrontier(op_set, shuffled_source_indices, self.k, self.j, self.seed, self.policy, self.priors)
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
            plan_stats = self._execute_sentinel_plan(plan, op_frontiers, expected_outputs, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing sentinel plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return plan_stats
