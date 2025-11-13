
import logging

import numpy as np
from chromadb.api.models.Collection import Collection

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import OperatorCostEstimates, OperatorStats, RecordOpStats, SentinelPlanStats
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter, NonLLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.progress import create_progress_manager
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)

# NOTE: we currently do not support Sentinel Plans with aggregates or limits which are not the final plan operator

class OpFrontier:
    """
    This class represents the set of operators which are currently in the frontier for a given logical operator.
    Each operator in the frontier is an instance of a PhysicalOperator which either:

    1. lies on the Pareto frontier of the set of sampled operators, or
    2. has been sampled fewer than j times
    """

    def __init__(
            self,
            op_set: list[PhysicalOperator],
            source_unique_logical_op_ids: list[str],
            root_dataset_ids: list[str],
            source_indices: list[tuple],
            k: int,
            j: int,
            seed: int,
            policy: Policy,
            priors: dict | None = None,
            dont_use_priors: bool = False,
        ):
        # set k and j, which are the initial number of operators in the frontier and the
        # initial number of records to sample for each frontier operator
        self.k = min(k, len(op_set))
        self.j = j
        self.source_indices = source_indices
        self.root_dataset_ids = root_dataset_ids
        self.dont_use_priors = dont_use_priors

        # store the policy that we are optimizing under
        self.policy = policy

        # store the prior beliefs on operator performance (if provided)
        self.priors = priors

        # boolean indication of the type of operator in this OpFrontier
        sample_op = op_set[0]
        self.is_scan_op = isinstance(sample_op, (ScanPhysicalOp, ContextScanOp))
        self.is_filter_op = isinstance(sample_op, FilterOp)
        self.is_aggregate_op = isinstance(sample_op, AggregateOp)
        self.is_llm_join = isinstance(sample_op, JoinOp)
        is_llm_convert = isinstance(sample_op, LLMConvert)
        is_llm_filter = isinstance(sample_op, LLMFilter)
        is_llm_topk = isinstance(sample_op, TopKOp) and isinstance(sample_op.index, Collection)
        self.is_llm_op = is_llm_convert or is_llm_filter or is_llm_topk or self.is_llm_join
        self.is_llm_convert = is_llm_convert

        # get order in which we will sample physical operators for this logical operator
        sample_op_indices = self._get_op_index_order(op_set, seed)

        # construct the initial set of frontier and reservoir operators
        self.frontier_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[:self.k]]
        self.reservoir_ops = [op_set[sample_idx] for sample_idx in sample_op_indices[self.k:]]
        self.off_frontier_ops: list[PhysicalOperator] = []

        # keep track of the source indices processed by each physical operator
        self.full_op_id_to_sources_processed = {op.get_full_op_id(): set() for op in op_set}
        self.full_op_id_to_sources_not_processed = {op.get_full_op_id(): source_indices for op in op_set}
        self.max_inputs = len(source_indices)

        # set the initial inputs for this logical operator; we maintain a mapping from source_unique_logical_op_id --> source_indices --> input;
        # for each unique source and (tuple of) source indices, we store its output, which is an input to this operator
        # for scan operators, we use the default name "source" since these operators have no source
        self.source_indices_to_inputs = {source_unique_logical_op_id: {} for source_unique_logical_op_id in source_unique_logical_op_ids}
        if self.is_scan_op:
            self.source_indices_to_inputs["source"] = {source_idx: [int(source_idx.split("-")[-1])] for source_idx in source_indices}
        

    def get_frontier_ops(self) -> list[PhysicalOperator]:
        """
        Returns the set of frontier operators for this OpFrontier.
        """
        return self.frontier_ops

    def get_off_frontier_ops(self) -> list[PhysicalOperator]:
        """
        Returns the set of off-frontier operators for this OpFrontier.
        """
        return self.off_frontier_ops

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

    def _compute_naive_priors(self, op_set: list[PhysicalOperator]) -> dict[str, dict[str, float]]:
        naive_priors = {}
        for op in op_set:
            # use naive cost estimates with dummy source estimates to compute priors
            source_op_estimates = OperatorCostEstimates(quality=1.0, cost_per_record=0.0, time_per_record=0.0, cardinality=100)
            op_estimates = (
                op.naive_cost_estimates(source_op_estimates, source_op_estimates)
                if self.is_llm_join
                else op.naive_cost_estimates(source_op_estimates)
            )

            # get op_id for this operator
            op_id = op.get_op_id()

            # set the naive quality, cost, and time priors for this operator
            naive_priors[op_id] = {
                "quality": op_estimates.quality,
                "cost": op_estimates.cost_per_record,
                "time": op_estimates.time_per_record,
            }

        return naive_priors

    def _get_op_index_order(self, op_set: list[PhysicalOperator], seed: int) -> list[int]:
        """
        Returns a list of indices for the operators in the op_set.
        """
        # if this is not an llm-operator, we simply return the indices in random order
        if not self.is_llm_op or self.dont_use_priors:
            if self.is_llm_convert:
                print("Using NO PRIORS for operator sampling order")
            rng = np.random.default_rng(seed=seed)
            op_indices = np.arange(len(op_set))
            rng.shuffle(op_indices)
            return op_indices

        # if this is an llm-operator, but we do not have priors, we first compute naive priors
        if self.priors is None or any([op_id not in self.priors for op_id in map(lambda op: op.get_op_id(), op_set)]):
            if self.is_llm_convert:
                print("Using NAIVE PRIORS for operator sampling order")
            self.priors = self._compute_naive_priors(op_set)

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

    def _get_op_source_indices_pairs(self) -> list[tuple[PhysicalOperator, tuple[str] | None]]:
        """
        Returns a list of tuples for (op, source_indices) which this operator needs to execute
        in the next iteration.
        """
        op_source_indices_pairs = []

        # if this operator is not being optimized: we don't request inputs, but simply process what we are given / told to (in the case of scans)
        if not self.is_llm_op and len(self.frontier_ops) == 1:
            return [(self.frontier_ops[0], None)]

        # otherwise, sample (operator, source_indices) pairs
        for op in self.frontier_ops:
            # execute new operators on first j indices per root dataset, and previously sampled operators on one per root dataset
            new_operator = self.full_op_id_to_sources_processed[op.get_full_op_id()] == set()
            samples_per_root_dataset = self.j if new_operator else 1
            num_root_datasets = len(self.root_dataset_ids)
            num_samples = samples_per_root_dataset**num_root_datasets
            samples = self.full_op_id_to_sources_not_processed[op.get_full_op_id()][:num_samples]
            for source_indices in samples:
                op_source_indices_pairs.append((op, source_indices))

        return op_source_indices_pairs

    def get_source_indices_for_next_iteration(self) -> set[tuple[str]]:
        """
        Returns the set of source indices which need to be sampled for the next iteration.
        """
        op_source_indices_pairs = self._get_op_source_indices_pairs()
        return set([source_indices for _, source_indices in op_source_indices_pairs if source_indices is not None])

    def get_frontier_op_inputs(self, source_indices_to_sample: set[tuple[str]], max_quality_op: PhysicalOperator) -> list[tuple[PhysicalOperator, tuple[str], list[DataRecord] | list[int] | None]]:
        """
        Returns the list of frontier operators and their next input to process. If there are
        any indices in `source_indices_to_sample` which this operator does not sample on its own, then
        we also have this frontier process those source indices' input with its max quality operator.
        """
        # if this is an aggregate, run on every input
        if self.is_aggregate_op:
            # NOTE: we don't keep track of source indices for aggregate (would require computing powerset of all source records);
            #       thus, we cannot currently support optimizing plans w/LLM operators after aggregations
            op = self.frontier_ops[0]
            all_inputs = []
            for _, source_indices_to_inputs in self.source_indices_to_inputs.items():
                for _, inputs in source_indices_to_inputs.items():
                    all_inputs.extend(inputs)
            return [(op, tuple(), all_inputs)]

        ### for optimized operators
        # get the list of (op, source_indices) pairs which this operator needs to execute
        op_source_indices_pairs = self._get_op_source_indices_pairs()

        # remove any root datasets which this op frontier does not have access to from the source_indices_to_sample
        def remove_unavailable_root_datasets(source_indices: str | tuple) -> str | tuple | None:
            # base case: source_indices is a string
            if isinstance(source_indices, str):
                return source_indices if source_indices.split("---")[0] in self.root_dataset_ids else None

            # recursive case: source_indices is a tuple
            left_indices = source_indices[0]
            right_indices = source_indices[1]
            left_filtered = remove_unavailable_root_datasets(left_indices)
            right_filtered = remove_unavailable_root_datasets(right_indices)
            if left_filtered is None and right_filtered is None:
                return None

            if left_filtered is None:
                return right_filtered
            if right_filtered is None:
                return left_filtered
            return (left_filtered, right_filtered)

        source_indices_to_sample = {remove_unavailable_root_datasets(source_indices) for source_indices in source_indices_to_sample}

        # if there are any source_indices in source_indices_to_sample which are not sampled by this operator,
        # apply the max quality operator (and any other frontier operators with no samples)
        sampled_source_indices = set(map(lambda tup: tup[1], op_source_indices_pairs))
        unsampled_source_indices = source_indices_to_sample - sampled_source_indices
        for source_indices in unsampled_source_indices:
            op_source_indices_pairs.append((max_quality_op, source_indices))
            for op in self.frontier_ops:
                if self.full_op_id_to_sources_processed[op.get_full_op_id()] == set() and op.get_full_op_id() != max_quality_op.get_full_op_id():
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

    def update_frontier(self, unique_logical_op_id: str, plan_stats: SentinelPlanStats, full_op_id_to_source_indices_processed: dict[str, set[list]]) -> None:
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
        full_op_id_to_op_stats: dict[str, OperatorStats] = plan_stats.operator_stats.get(unique_logical_op_id, {})
        full_op_id_to_record_op_stats: dict[str, list[RecordOpStats]] = {}
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

        # NOTE: it is possible for the full_op_id_to_record_op_stats to be empty if there is a duplicate operator
        # (e.g. a scan of the same dataset) which has all of its results cached and no new_record_op_stats;
        # in this case, we do not update the frontier
        if full_op_id_to_record_op_stats == {}:
            return

        # update the set of source indices processed by each physical operator
        for full_op_id, source_indices_processed in full_op_id_to_source_indices_processed.items():
            # update the set of source indices processed
            for source_indices in source_indices_processed:
                source_indices = source_indices[0] if len(source_indices) == 1 else tuple(source_indices)
                self.full_op_id_to_sources_processed[full_op_id].add(source_indices)
                if source_indices in self.full_op_id_to_sources_not_processed[full_op_id]:
                    self.full_op_id_to_sources_not_processed[full_op_id].remove(source_indices)

            # update the set of source indices not processed
            self.full_op_id_to_sources_not_processed[full_op_id] = [
                indices for indices in self.full_op_id_to_sources_not_processed[full_op_id]
                if indices not in source_indices_processed
            ]

        # compute mapping of physical op to num samples and total samples drawn
        full_op_id_to_num_samples, total_num_samples = {}, 0
        for full_op_id, record_op_stats_lst in full_op_id_to_record_op_stats.items():
            # compute the number of samples as the length of the record_op_stats_lst
            num_samples = len(record_op_stats_lst)
            full_op_id_to_num_samples[full_op_id] = num_samples
            total_num_samples += num_samples

        # compute avg. selectivity, cost, time, and quality for each physical operator
        def total_output(record_op_stats_lst: list[RecordOpStats]):
            return sum([record_op_stats.passed_operator for record_op_stats in record_op_stats_lst])

        def total_input(record_op_stats_lst: list[RecordOpStats]):
            # TODO: this is okay for now because we only really need these calculations for Converts and Filters,
            #       but this will need more thought if/when we optimize joins
            all_parent_ids = []
            for record_op_stats in record_op_stats_lst:
                all_parent_ids.extend(
                    [None]
                    if record_op_stats.record_parent_ids is None
                    else record_op_stats.record_parent_ids
                )
            return len(set(all_parent_ids))

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
            full_op_id: np.mean([record_op_stats.quality for record_op_stats in record_op_stats_lst if record_op_stats.quality is not None])
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
            if len(self.full_op_id_to_sources_processed[full_op_id]) == self.max_inputs:
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
            max_quality_record, max_quality = records_lst[0], record_op_stats_lst[0].quality if record_op_stats_lst[0].quality is not None else 0.0
            max_quality_stats = record_op_stats_lst[0]
            for record, record_op_stats in zip(records_lst[1:], record_op_stats_lst[1:]):
                record_quality = record_op_stats.quality if record_op_stats.quality is not None else 0.0
                if record_quality > max_quality:
                    max_quality_record = record
                    max_quality = record_quality
                    max_quality_stats = record_op_stats
            out_records.append(max_quality_record)
            out_record_op_stats.append(max_quality_stats)

        # create and return final DataRecordSet
        return DataRecordSet(out_records, out_record_op_stats)

    def update_inputs(self, source_unique_logical_op_id: str, source_indices_to_record_sets: dict[tuple[int], list[DataRecordSet]]):
        """
        Update the inputs for this logical operator based on the outputs of the previous logical operator.
        """
        for source_indices, record_sets in source_indices_to_record_sets.items():
            input = []
            max_quality_record_set = self.pick_highest_quality_output(record_sets)
            for record in max_quality_record_set:
                input.append(record if record._passed_operator else None)

            self.source_indices_to_inputs[source_unique_logical_op_id][source_indices] = input

class MABExecutionStrategy(SentinelExecutionStrategy):
    """
    This class implements the Multi-Armed Bandit (MAB) execution strategy for SentinelQueryProcessors.

    NOTE: the number of samples will slightly exceed the sample_budget if the number of operator
    calls does not perfectly match the sample_budget. This may cause some minor discrepancies with
    the progress manager as a result.
    """
    def _remove_filtered_records_from_downstream_ops(self, topo_idx: int, plan: SentinelPlan, op_frontiers: dict[str, OpFrontier], source_indices_to_all_record_sets: dict[int, list[DataRecordSet]]) -> None:
        """Remove records which were filtered out by a NonLLMFilter from all downstream operators."""
        filtered_source_indices = set()

        # NonLLMFilter will have one record_set per source_indices with a single record
        for source_indices, record_sets in source_indices_to_all_record_sets.items():
            record: DataRecord = record_sets[0][0]
            if not record._passed_operator:
                filtered_source_indices.add(source_indices)

        # remove filtered source indices from all downstream operators
        if len(filtered_source_indices) > 0:
            for downstream_topo_idx in range(topo_idx + 1, len(plan)):
                downstream_logical_op_id = plan[downstream_topo_idx][0]
                downstream_unique_logical_op_id = f"{downstream_topo_idx}-{downstream_logical_op_id}"
                downstream_op_frontier = op_frontiers[downstream_unique_logical_op_id]
                for full_op_id in downstream_op_frontier.full_op_id_to_sources_not_processed:
                    downstream_op_frontier.full_op_id_to_sources_not_processed[full_op_id] = [
                        indices for indices in downstream_op_frontier.full_op_id_to_sources_not_processed[full_op_id]
                        if indices not in filtered_source_indices
                    ]

    def _get_max_quality_op(self, unique_logical_op_id: str, op_frontiers: dict[str, OpFrontier], plan_stats: SentinelPlanStats) -> PhysicalOperator:
        """
        Returns the operator in the frontier with the highest (estimated) quality.
        """
        # get the (off) frontier operators for this logical_op_id
        frontier_ops = op_frontiers[unique_logical_op_id].get_frontier_ops() + op_frontiers[unique_logical_op_id].get_off_frontier_ops()

        # get a mapping from full_op_id --> list[RecordOpStats]
        full_op_id_to_op_stats: dict[str, OperatorStats] = plan_stats.operator_stats.get(unique_logical_op_id, {})
        full_op_id_to_record_op_stats = {
            full_op_id: op_stats.record_op_stats_lst
            for full_op_id, op_stats in full_op_id_to_op_stats.items()
        }

        # iterate over the frontier ops and return the one with the highest quality
        max_quality_op, max_avg_quality = None, None
        for op in frontier_ops:
            op_quality_stats = []
            if op.get_full_op_id() in full_op_id_to_record_op_stats:
                op_quality_stats = [
                    record_op_stats.quality
                    for record_op_stats in full_op_id_to_record_op_stats[op.get_full_op_id()]
                    if record_op_stats.quality is not None
                ]
            avg_op_quality = sum(op_quality_stats) / len(op_quality_stats) if len(op_quality_stats) > 0 else 0.0
            if max_avg_quality is None or avg_op_quality > max_avg_quality:
                max_quality_op = op
                max_avg_quality = avg_op_quality

        return max_quality_op

    def _compute_termination_condition(self, samples_drawn: int, sampling_cost: float) -> bool:
        return (samples_drawn >= self.sample_budget) if self.sample_cost_budget is None else (sampling_cost >= self.sample_cost_budget)

    def _execute_sentinel_plan(
            self,
            plan: SentinelPlan,
            op_frontiers: dict[str, OpFrontier],
            validator: Validator,
            plan_stats: SentinelPlanStats,
        ) -> SentinelPlanStats:
        # sample records and operators and update the frontiers
        samples_drawn, sampling_cost = 0, 0.0
        while not self._compute_termination_condition(samples_drawn, sampling_cost):
            # pre-compute the set of source indices which will need to be sampled
            source_indices_to_sample = set()
            for op_frontier in op_frontiers.values():
                source_indices = op_frontier.get_source_indices_for_next_iteration()
                source_indices_to_sample.update(source_indices)

            # execute operator sets in sequence
            for topo_idx, (logical_op_id, op_set) in enumerate(plan):
                # compute unique logical op id within plan
                unique_logical_op_id = f"{topo_idx}-{logical_op_id}"

                # use the execution cache to determine the maximum quality operator for this logical_op_id
                max_quality_op = self._get_max_quality_op(unique_logical_op_id, op_frontiers, plan_stats)

                # get frontier ops and their next input
                def filter_and_clean_inputs(frontier_op_inputs: list[tuple]) -> bool:
                    cleaned_inputs = []
                    for tup in frontier_op_inputs:
                        input = tup[-1]
                        if isinstance(input, list):
                            input = [record for record in input if record is not None]
                        if input is not None and input != []:
                            cleaned_inputs.append((tup[0], tup[1], input))
                    return cleaned_inputs
                frontier_op_inputs = op_frontiers[unique_logical_op_id].get_frontier_op_inputs(source_indices_to_sample, max_quality_op)
                frontier_op_inputs = filter_and_clean_inputs(frontier_op_inputs)

                # break out of the loop if frontier_op_inputs is empty, as this means all records have been filtered out
                if len(frontier_op_inputs) == 0:
                    continue

                # run sampled operators on sampled inputs and update the number of samples drawn
                source_indices_to_record_set_tuples, num_llm_ops = self._execute_op_set(unique_logical_op_id, frontier_op_inputs)
                samples_drawn += num_llm_ops

                # score the quality of each generated output
                source_indices_to_all_record_sets = {
                    source_indices: [(record_set, op) for record_set, op, _ in record_set_tuples]
                    for source_indices, record_set_tuples in source_indices_to_record_set_tuples.items()
                }
                source_indices_to_all_record_sets, val_gen_stats = self._score_quality(validator, source_indices_to_all_record_sets)

                # update the progress manager with validation cost
                self.progress_manager.incr_overall_progress_cost(val_gen_stats.cost_per_record)

                # remove records that were read from the execution cache before adding to record op stats
                new_record_op_stats = []
                for _, record_set_tuples in source_indices_to_record_set_tuples.items():
                    for record_set, _, is_new in record_set_tuples:
                        if is_new:
                            new_record_op_stats.extend(record_set.record_op_stats)

                # update plan stats
                plan_stats.add_record_op_stats(unique_logical_op_id, new_record_op_stats)
                plan_stats.add_validation_gen_stats(unique_logical_op_id, val_gen_stats)
                sampling_cost = plan_stats.get_total_cost_so_far()

                # provide the best record sets as inputs to the next logical operator
                next_unique_logical_op_id = plan.get_next_unique_logical_op_id(unique_logical_op_id)
                if next_unique_logical_op_id is not None:
                    source_indices_to_all_record_sets = {
                        source_indices: [record_set for record_set, _ in record_set_tuples]
                        for source_indices, record_set_tuples in source_indices_to_all_record_sets.items()
                    }
                    op_frontiers[next_unique_logical_op_id].update_inputs(unique_logical_op_id, source_indices_to_all_record_sets)

                # update the (pareto) frontier for each set of operators
                full_op_id_to_source_indices_processed = {}
                for source_indices, record_set_tuples in source_indices_to_record_set_tuples.items():
                    for _, op, _ in record_set_tuples:
                        if op.get_full_op_id() not in full_op_id_to_source_indices_processed:
                            full_op_id_to_source_indices_processed[op.get_full_op_id()] = set()
                        full_op_id_to_source_indices_processed[op.get_full_op_id()].add(source_indices)
                op_frontiers[unique_logical_op_id].update_frontier(unique_logical_op_id, plan_stats, full_op_id_to_source_indices_processed)

                # if the operator is a non-llm filter which has filtered out records, remove those records from
                # all downstream operators' full_op_id_to_sources_not_processed
                if isinstance(op_set[0], NonLLMFilter) and next_unique_logical_op_id is not None:
                    self._remove_filtered_records_from_downstream_ops(topo_idx, plan, op_frontiers, source_indices_to_all_record_sets)

        # finalize plan stats
        plan_stats.finish()

        return plan_stats


    def execute_sentinel_plan(self, plan: SentinelPlan, train_dataset: dict[str, Dataset], validator: Validator) -> SentinelPlanStats:
        logger.info(f"Executing plan {plan.plan_id} with {self.max_workers} workers")

        # initialize plan stats
        plan_stats = SentinelPlanStats.from_plan(plan)
        plan_stats.start()

        # shuffle the indices of records to sample
        dataset_id_to_shuffled_source_indices = {}
        for dataset_id, dataset in train_dataset.items():
            total_num_samples = len(dataset)
            shuffled_source_indices = [f"{dataset_id}---{int(idx)}" for idx in np.arange(total_num_samples)]
            self.rng.shuffle(shuffled_source_indices)
            dataset_id_to_shuffled_source_indices[dataset_id] = shuffled_source_indices

        # initialize frontier for each logical operator
        op_frontiers = {}
        for topo_idx, (logical_op_id, op_set) in enumerate(plan):
            unique_logical_op_id = f"{topo_idx}-{logical_op_id}"
            source_unique_logical_op_ids = plan.get_source_unique_logical_op_ids(unique_logical_op_id)
            root_dataset_ids = plan.get_root_dataset_ids(unique_logical_op_id)
            sample_op = op_set[0]
            if isinstance(sample_op, (ScanPhysicalOp, ContextScanOp)):
                assert len(root_dataset_ids) == 1, f"Scan for {sample_op} has {len(root_dataset_ids)} > 1 root dataset ids"
                root_dataset_id = root_dataset_ids[0]
                source_indices = dataset_id_to_shuffled_source_indices[root_dataset_id]
                op_frontiers[unique_logical_op_id] = OpFrontier(op_set, source_unique_logical_op_ids, root_dataset_ids, source_indices, self.k, self.j, self.seed, self.policy, self.priors, self.dont_use_priors)
            elif isinstance(sample_op, JoinOp):
                assert len(source_unique_logical_op_ids) == 2, f"Join for {sample_op} has {len(source_unique_logical_op_ids)} != 2 source logical operators"
                left_source_indices = op_frontiers[source_unique_logical_op_ids[0]].source_indices
                right_source_indices = op_frontiers[source_unique_logical_op_ids[1]].source_indices
                source_indices = []
                for left_source_idx in left_source_indices:
                    for right_source_idx in right_source_indices:
                        source_indices.append((left_source_idx, right_source_idx))
                op_frontiers[unique_logical_op_id] = OpFrontier(op_set, source_unique_logical_op_ids, root_dataset_ids, source_indices, self.k, self.j, self.seed, self.policy, self.priors, self.dont_use_priors)
            else:
                source_indices = op_frontiers[source_unique_logical_op_ids[0]].source_indices
                op_frontiers[unique_logical_op_id] = OpFrontier(op_set, source_unique_logical_op_ids, root_dataset_ids, source_indices, self.k, self.j, self.seed, self.policy, self.priors, self.dont_use_priors)

        # initialize and start the progress manager
        self.progress_manager = create_progress_manager(plan, sample_budget=self.sample_budget, sample_cost_budget=self.sample_cost_budget, progress=self.progress)
        self.progress_manager.start()

        # NOTE: we must handle progress manager outside of _execute_sentinel_plan to ensure that it is shut down correctly;
        #       if we don't have the `finally:` branch, then program crashes can cause future program runs to fail because
        #       the progress manager cannot get a handle to the console 
        try:
            # execute sentinel plan by sampling records and operators
            plan_stats = self._execute_sentinel_plan(plan, op_frontiers, validator, plan_stats)

        finally:
            # finish progress tracking
            self.progress_manager.finish()

        logger.info(f"Done executing sentinel plan: {plan.plan_id}")
        logger.debug(f"Plan stats: (plan_cost={plan_stats.total_plan_cost}, plan_time={plan_stats.total_plan_time})")

        return plan_stats
