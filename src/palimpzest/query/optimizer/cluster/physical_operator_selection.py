"""Selection policy for sampling physical operators from cluster trees."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from palimpzest.core.models import OperatorCostEstimates, PlanCost
from palimpzest.policy import MaxQuality, MinCost, MinTime, Policy
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.optimizer.cluster.physical_operator_clustering import (
    PhysicalOperatorCluster,
)


@dataclass
class PhysicalOperatorSelection:
    """A sampled physical operator and the cluster branch that produced it.

    The cluster_path is a list of clusters from the root down to the leaf
    that contains the selected physical operator. The leaf cluster is the last
    element; internal clusters are its ancestors. The path is used during
    record_execution() to update sample counts and propagate new cost
    estimates up the tree.
    """

    record_id: str | int
    physical_op: PhysicalOperator
    cluster_path: list[PhysicalOperatorCluster]

class PhysicalOperatorSelector:
    """Stateful sampler for the clustered physical operator search space.

    The selector owns sampling state that changes across rounds: how many times
    each cluster branch has been sampled, and which physical operators have
    already been executed for each input record. It chooses a branch using the
    exploration/exploitation score, then records executions back into the
    affected cluster path.

    The physical operator selector keeps a running state of the physical ops that have been sampled and their observed estimates/qualities.

    Key mutable state:
      - cluster_sample_counts: total number of times each cluster (identified by id) has been visited during sampling.
      - cluster_input_record_counts: number of input records considered for each cluster, used as a normalizer for confidence.
      - physical_op_sample_counts: how many times each concrete physical operator (by full op id) has been sampled.
      - executed_physical_op_ids_by_record: per-record set of physical operator ids already executed (to avoid re-selection).
      - physical_op_quality_samples: list of observed quality values per physical operator (maintained during record_execution).

    The selector uses a tree of PhysicalOperatorCluster nodes. Every cluster's
    physical_ops lists the concrete operators under that subtree. A *leaf
    cluster* owns physical_op_cost_estimates for its concrete operators; an
    *internal cluster* aggregates child cost_estimates. The cluster_path
    in a PhysicalOperatorSelection tracks the branch from root to leaf that
    produced a chosen operator.

    After all sampling rounds, select_best_physical_operator() performs a final
    exploitation step by evaluating all leaf operators with uncertainty_aware_select.
    """

    def __init__(
        self,
        policy: Policy,
        total_sampling_rounds: int,
        max_input_records: int = 1,
        seed: int = 42,
    ):
        """Create a selector for a fixed policy and sampling budget.

        max_input_records is used as the default confidence normalizer when
        a caller selects a physical operator for an already-chosen record.
        """
        self.policy = policy
        self.total_sampling_rounds = total_sampling_rounds
        self.max_input_records = max_input_records
        self.rng = random.Random(seed)
        self.cluster_sample_counts = defaultdict(int)
        self.cluster_input_record_counts = defaultdict(lambda: self.max_input_records)
        self.physical_op_sample_counts = defaultdict(int)
        self.executed_physical_op_ids_by_record = defaultdict(set)

        self.physical_op_quality_samples = defaultdict(list)

    def uncertainty_aware_select(
        self,
        physical_op_cost_estimates: dict[str, OperatorCostEstimates],
        physical_op_confidences: dict[str, float],
        uncertainty_weight: float = 0.0,
    ) -> tuple[str, OperatorCostEstimates]:
        """Return the physical operator id with the best final score.
        The score for each operator is 
            policy_score - uncertainty_weight * (1 - confidence)
        Policy scores are normalized to [0,1] based on the best and worst metric
        among the candidates. In case of a tie, the policy's choose() method
        breaks it.

        This strategy is used by PhysicalOperatorSelector.select_best_physical_operator()
        after all sampling rounds are complete. It combines the raw policy score
        (based on quality, cost, or time) with an uncertainty penalty derived from
        the confidence (fraction of samples collected) for each operator. The final
        selection picks the operator with the highest composite score.

        Supported policies: MaxQuality, MinCost, MinTime.
        """

        if len(physical_op_cost_estimates) == 0:
            raise ValueError("Cannot select from an empty physical operator set")
        weight = max(uncertainty_weight, 0.0)

        metric_values = {}
        for physical_op_id, cost_estimates in physical_op_cost_estimates.items():
            if isinstance(self.policy, MaxQuality):
                metric_values[physical_op_id] = cost_estimates.quality
            elif isinstance(self.policy, MinCost):
                metric_values[physical_op_id] = cost_estimates.cost_per_record
            elif isinstance(self.policy, MinTime):
                metric_values[physical_op_id] = cost_estimates.time_per_record

        min_value = min(metric_values.values())
        max_value = max(metric_values.values())

        best_physical_op_id = None
        best_cost_estimates = None
        best_score = None
        for physical_op_id, cost_estimates in physical_op_cost_estimates.items():
            value = metric_values[physical_op_id]
            if min_value == max_value:
                policy_score = 1.0
            elif isinstance(self.policy, MaxQuality):
                policy_score = (value - min_value) / (max_value - min_value)
            else:
                policy_score = (max_value - value) / (max_value - min_value)

            confidence = physical_op_confidences.get(physical_op_id, 0.0)
            uncertainty = 1.0 - max(0.0, min(confidence, 1.0))
            score = policy_score - (weight * uncertainty)
            if best_score is None or score > best_score:
                best_physical_op_id = physical_op_id
                best_cost_estimates = cost_estimates
                best_score = score
            elif score == best_score:
                plan_cost = PlanCost(
                    cost=cost_estimates.cost_per_record,
                    time=cost_estimates.time_per_record,
                    quality=cost_estimates.quality,
                )
                assert best_cost_estimates is not None
                best_plan_cost = PlanCost(
                    cost=best_cost_estimates.cost_per_record,
                    time=best_cost_estimates.time_per_record,
                    quality=best_cost_estimates.quality,
                )
                if self.policy.choose(plan_cost, best_plan_cost):
                    best_physical_op_id = physical_op_id
                    best_cost_estimates = cost_estimates

        assert best_physical_op_id is not None and best_cost_estimates is not None
        return best_physical_op_id, best_cost_estimates

    def select_next_sample(
        self,
        root_cluster: PhysicalOperatorCluster,
        input_record_ids: list[str | int],
        sampling_round_idx: int,
    ) -> PhysicalOperatorSelection:
        """Choose both a random input record and a physical operator for it.

        Records for which every physical operator has already been executed are
        excluded before sampling the record.  If the root cluster contains
        exactly one physical operator, that operator is returned directly with a
        cluster_path that walks down to its leaf (following children that contain
        the same operator id).
        """
        available_record_ids = [
            record_id
            for record_id in input_record_ids
            if self._cluster_has_available_physical_op(root_cluster, record_id)
        ]
        if len(available_record_ids) == 0:
            raise ValueError(
                f"All physical operators in cluster {root_cluster.name} have already "
                "been executed on every candidate input record"
            )
        record_id = self.rng.choice(input_record_ids)

        # If the root cluster only has one physical operator, skip selection but
        # preserve the path to the leaf that owns the operator cost estimates.
        if len(root_cluster.physical_ops) == 1:
            physical_op = root_cluster.physical_ops[0]
            physical_op_id = physical_op.get_full_op_id()
            cluster = root_cluster
            cluster_path = [root_cluster]
            while len(cluster.children) > 0:
                matching_children = [
                    child
                    for child in cluster.children
                    if any(
                        op.get_full_op_id() == physical_op_id
                        for op in child.physical_ops
                    )
                ]
                cluster = matching_children[0]
                cluster_path.append(cluster)

            return PhysicalOperatorSelection(
                record_id=record_id,
                physical_op=physical_op,
                cluster_path=cluster_path,
            )

        return self.select_physical_operator(
            root_cluster,
            record_id,
            sampling_round_idx,
            max_input_records=len(input_record_ids),
        )

    def select_physical_operator(
        self,
        root_cluster: PhysicalOperatorCluster,
        record_id: str | int,
        sampling_round_idx: int,
        max_input_records: int | None = None,
    ) -> PhysicalOperatorSelection:
        """Choose a physical operator by walking from root cluster to leaf.

        At each internal node, child weights combine uncertainty and policy
        score: alpha * (1 - confidence) + (1 - alpha) * predicted_score.
        alpha decreases as the sampling budget is consumed.

        The walking stops when a leaf cluster is reached (no children). Among
        the leaf's available physical operators (those not yet executed for this
        record), one is chosen uniformly at random.
        """
        if not self._cluster_has_available_physical_op(root_cluster, record_id):
            raise ValueError(
                f"All physical operators in cluster {root_cluster.name} have already "
                f"been executed on record {record_id}"
            )

        alpha = (self.total_sampling_rounds - sampling_round_idx - 1) / max(
            self.total_sampling_rounds,
            1,
        )
        alpha = max(0.0, min(1.0, alpha))
        if max_input_records is not None:
            self.cluster_input_record_counts[id(root_cluster)] = max_input_records

        cluster = root_cluster
        cluster_path = [root_cluster]
        while len(cluster.children) > 0:
            available_children = [
                child
                for child in cluster.children
                if self._cluster_has_available_physical_op(child, record_id)
            ]
            policy_scores = self._policy_scores(available_children)
            weights = []
            for child in available_children:
                confidence = self.cluster_confidence_entropy(
                    child,
                    max_input_records=max_input_records,
                )
                predicted_score = policy_scores[id(child)]
                weights.append(
                    alpha * (1.0 - confidence)
                    + (1.0 - alpha) * predicted_score
                )

            cluster = self._weighted_choice(available_children, weights)
            cluster_path.append(cluster)

        available_physical_ops = [
            op
            for op in cluster.physical_ops
            if op.get_full_op_id()
            not in self.executed_physical_op_ids_by_record[record_id]
        ]
        physical_op = self.rng.choice(available_physical_ops)
        return PhysicalOperatorSelection(
            record_id=record_id,
            physical_op=physical_op,
            cluster_path=cluster_path,
        )

    def record_execution(
        self,
        selection: PhysicalOperatorSelection,
        cost_estimates: OperatorCostEstimates | None = None,
    ) -> None:
        """Record an execution and refresh scores for the affected branch.

        The selected physical operator is blacklisted for the selected record,
        every cluster on the selected path gets one additional sample, and the
        path's cost estimates are recomputed from leaf to root.

        If cost_estimates is None, the leaf cluster's stored estimates for
        that operator are used.  The observed quality value is appended to
        physical_op_quality_samples for later entropy-based confidence
        calculations.
        """
        physical_op_id = selection.physical_op.get_full_op_id()
        self.executed_physical_op_ids_by_record[selection.record_id].add(physical_op_id)
        self.physical_op_sample_counts[physical_op_id] += 1

        leaf_cluster = selection.cluster_path[-1]
        if cost_estimates is None:
            cost_estimates = leaf_cluster.physical_op_cost_estimates[physical_op_id]
        leaf_cluster.physical_op_cost_estimates[physical_op_id] = cost_estimates

        for cluster in selection.cluster_path:
            self.cluster_sample_counts[id(cluster)] += 1

        # Propagate new aggregated cost estimates up the tree: leaves average
        # their own operator estimates, internal clusters average children's.
        for cluster in reversed(selection.cluster_path):
            if len(cluster.children) == 0:
                cluster.cost_estimates = self._average_cost_estimates(
                    list(cluster.physical_op_cost_estimates.values())
                )
            else:
                cluster.cost_estimates = self._average_cost_estimates(
                    [
                        child.cost_estimates
                        for child in cluster.children
                        if child.cost_estimates is not None
                    ]
                )

        self.physical_op_quality_samples[physical_op_id].append(cost_estimates.quality)

    def select_best_physical_operator(
        self,
        root_cluster: PhysicalOperatorCluster,
    ) -> tuple[PhysicalOperator, OperatorCostEstimates]:
        """Return the best concrete physical operator under a cluster.

        This is the final exploitation step after sampling has updated the
        cluster tree.  It collects all leaf operators' cost estimates,
        computes confidences as the fraction of maximum possible samples
        collected per operator, and delegates to uncertainty_aware_select() to pick the best operator.

        Only single-objective policies (MaxQuality, MinCost, MinTime)
        are currently supported.
        """
        if not isinstance(self.policy, (MaxQuality, MinCost, MinTime)):
            raise NotImplementedError(
                f"Unsupported physical operator selection policy: {type(self.policy).__name__}"
            )

        physical_op_by_id = {
            op.get_full_op_id(): op
            for op in root_cluster.physical_ops
        }
        physical_op_cost_estimates = {}
        stack = [root_cluster]
        while len(stack) > 0:
            cluster = stack.pop()
            if len(cluster.children) == 0:
                physical_op_cost_estimates.update(cluster.physical_op_cost_estimates)
            else:
                stack.extend(cluster.children)

        if len(physical_op_cost_estimates) == 0:
            raise ValueError(f"Cluster {root_cluster.name} has no physical operators")

        max_input_records = self.cluster_input_record_counts[id(root_cluster)]
        physical_op_confidences = {
            physical_op_id: min(
                self.physical_op_sample_counts[physical_op_id]
                / max(max_input_records, 1),
                1.0,
            )
            for physical_op_id in physical_op_cost_estimates
        }
        best_physical_op_id, best_cost_estimates = self.uncertainty_aware_select(
            physical_op_cost_estimates,
            physical_op_confidences,
        )

        return physical_op_by_id[best_physical_op_id], best_cost_estimates

    def cluster_confidence(
        self,
        cluster: PhysicalOperatorCluster,
        max_input_records: int | None = None,
    ) -> float:
        """Return normalized confidence for a cluster in [0, 1].

        The normalizer is num_input_records * num_physical_ops_under_cluster.
        Confidence is based on the ratio of samples taken to the maximum
        possible (assuming all operators could be sampled for every record).
        """
        input_record_count = (
            self.max_input_records
            if max_input_records is None
            else max_input_records
        )
        max_possible_samples = max(input_record_count * len(cluster.physical_ops), 1)
        sample_count = self.cluster_sample_counts[id(cluster)]
        return min(sample_count / max_possible_samples, 1.0)

    def cluster_confidence_entropy(
        self,
        cluster: PhysicalOperatorCluster,
        max_input_records: int | None = None,
    ) -> float:
        """
        Return a confidence score which is based on the entropy of results obtained from the cluster.
        Clusters with a low entropy (i.e., more consistent results) will have a higher confidence score, while clusters with a high entropy (i.e., more varied results) will have a lower confidence score.
        The normalizer is num_input_records * num_physical_ops_under_cluster.

        NOTE: This method is not fully implemented. The breakpoint below marks
        the point where entropy calculation should be added. Currently it
        returns 0.0.
        """

        input_record_count = (
            self.max_input_records
            if max_input_records is None
            else max_input_records
        )
        max_possible_samples = max(input_record_count * len(cluster.physical_ops), 1)
        entropy = 0.0

        breakpoint()
        score = entropy

        return score

    def _cluster_has_available_physical_op(
        self,
        cluster: PhysicalOperatorCluster,
        record_id: str | int,
    ) -> bool:
        """Return whether a cluster has at least one unexecuted op for a record."""
        executed_physical_op_ids = self.executed_physical_op_ids_by_record[record_id]
        return any(
            op.get_full_op_id() not in executed_physical_op_ids
            for op in cluster.physical_ops
        )

    def _policy_scores(
        self,
        clusters: list[PhysicalOperatorCluster],
    ) -> dict[int, float]:
        """Normalize sibling cluster predictions according to the active policy.

        MaxQuality treats higher quality as better. MinCost and
        MinTime invert their metric so lower values receive higher scores.
        Normalization maps the best sibling to 1.0 and the worst to 0.0; if all
        siblings have the same value, all receive 1.0.
        """
        if not isinstance(self.policy, (MaxQuality, MinCost, MinTime)):
            raise NotImplementedError(
                f"Unsupported physical operator selection policy: {type(self.policy).__name__}"
            )

        cluster_values = {}
        for cluster in clusters:
            if cluster.cost_estimates is None:
                cluster_values[id(cluster)] = 0.0
            elif isinstance(self.policy, MaxQuality):
                cluster_values[id(cluster)] = cluster.cost_estimates.quality
            elif isinstance(self.policy, MinCost):
                cluster_values[id(cluster)] = cluster.cost_estimates.cost_per_record
            elif isinstance(self.policy, MinTime):
                cluster_values[id(cluster)] = cluster.cost_estimates.time_per_record

        min_value = min(cluster_values.values())
        max_value = max(cluster_values.values())
        if min_value == max_value:
            return {cluster_id: 1.0 for cluster_id in cluster_values}

        if isinstance(self.policy, MaxQuality):
            return {
                cluster_id: (value - min_value) / (max_value - min_value)
                for cluster_id, value in cluster_values.items()
            }

        return {
            cluster_id: (max_value - value) / (max_value - min_value)
            for cluster_id, value in cluster_values.items()
        }

    def _weighted_choice(
        self,
        clusters: list[PhysicalOperatorCluster],
        weights: list[float],
    ) -> PhysicalOperatorCluster:
        """Sample a cluster proportionally to non-negative weights."""
        total_weight = sum(weights)
        if total_weight <= 0:
            return self.rng.choice(clusters)

        threshold = self.rng.random() * total_weight
        running_weight = 0.0
        for cluster, weight in zip(clusters, weights, strict=True):
            running_weight += weight
            if running_weight >= threshold:
                return cluster

        return clusters[-1]

    def _average_cost_estimates(
        self,
        cost_estimates: list[OperatorCostEstimates],
    ) -> OperatorCostEstimates | None:
        """Average every populated OperatorCostEstimates field independently.

        Fields that are None in all estimates are left as None in the
        result.  If the list is empty, returns None.
        """
        if len(cost_estimates) == 0:
            return None

        averaged_fields = {}
        for field_name in OperatorCostEstimates.model_fields:
            values = [
                getattr(cost_estimate, field_name)
                for cost_estimate in cost_estimates
                if getattr(cost_estimate, field_name) is not None
            ]
            averaged_fields[field_name] = (
                sum(values) / len(values)
                if len(values) > 0
                else None
            )

        return OperatorCostEstimates(**averaged_fields)
