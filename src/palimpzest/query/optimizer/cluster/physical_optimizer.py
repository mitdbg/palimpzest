from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from palimpzest.constants import Cardinality
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import (
    GenerationStats,
    OperatorCostEstimates,
    OperatorStats,
    PlanCost,
    SentinelPlanStats,
)
from palimpzest.query.operators.aggregate import AggregateOp
from palimpzest.query.operators.batched import BatchedOperator
from palimpzest.query.operators.convert import ConvertOp, LLMConvert
from palimpzest.query.operators.filter import FilterOp, LLMFilter
from palimpzest.query.operators.join import JoinOp as PhysicalJoinOp
from palimpzest.query.operators.logical import BaseScan, ContextScan
from palimpzest.query.operators.logical import JoinOp as LogicalJoinOp
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.scan import ContextScanOp, ScanPhysicalOp
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer_config import OptimizerConfig
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan
from palimpzest.query.optimizer.cluster.physical_operator_clustering import (
    ManualPhysicalOperatorClusteringStrategy,
    PhysicalOperatorCluster,
    PhysicalOperatorClusteringStrategy,
)
from palimpzest.query.optimizer.cluster.physical_operator_selection import (
    PhysicalOperatorSelector,
)
from palimpzest.query.optimizer.cluster.physical_operator_registry import (
    build_physical_operator_candidates,
)
from palimpzest.query.plan import PhysicalPlan, SentinelPlan
from palimpzest.utils.progress import ProgressManager, create_progress_manager
from palimpzest.validator.validator import Validator


class PhysicalOptimizer:
    """Physical optimizer for clustered candidate search.

    This optimizer currently builds the physical candidate clusters and the
    stateful selector that will sample them. Candidate instantiation, clustering,
    and selection are kept as separate pieces so each can evolve independently.
    """

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        clustering_strategy: PhysicalOperatorClusteringStrategy | None = None,
        max_workers: int | None = 64,
        progress: bool = True,
    ):
        """Initialize optimizer state from runtime optimizer configuration."""
        self.optimizer_config = optimizer_config
        self.policy = optimizer_config.policy
        self.max_workers = max(max_workers or 64, 1)
        self.progress = progress
        self.clustering_strategy = (
            ManualPhysicalOperatorClusteringStrategy()
            if clustering_strategy is None
            else clustering_strategy
        )

    def build_physical_operator_clusters(
        self,
        logical_plan: LogicalPlan,
    ) -> tuple[
        dict[str, PhysicalOperatorCluster],
        dict[str, list[str]],
        list[str],
        dict[str, OperatorCostEstimates],
        dict[str, OperatorCostEstimates | None],
    ]:
        """Build physical clusters in source-to-sink logical plan order.

        The method first instantiates flat candidates for every logical
        operator. It then topologically walks the logical plan from source scans
        to downstream consumers, passing each logical operator's source cluster
        cost estimates into the clustering strategy. Base scans seed their
        source estimate from ``len(datasource)``; context scans seed cardinality
        ``1.0``; joins receive left and right source estimates.
        """
        op_candidates = build_physical_operator_candidates(
            logical_plan,
            self.optimizer_config,
        )

        source_op_ids = {op_id: [] for op_id in logical_plan.operators}
        for upstream_op_id, downstream_op_ids in logical_plan.edges.items():
            source_op_ids.setdefault(upstream_op_id, [])
            for downstream_op_id in downstream_op_ids:
                source_op_ids.setdefault(downstream_op_id, [])
                source_op_ids[downstream_op_id].append(upstream_op_id)

        remaining_source_counts = {
            op_id: len(source_ids) for op_id, source_ids in source_op_ids.items()
        }
        ready_op_ids = sorted(
            op_id for op_id, count in remaining_source_counts.items() if count == 0
        )
        op_clusters = {}
        op_cost_estimates = {}
        topological_order = []
        op_source_cost_estimates = {}
        op_right_source_cost_estimates = {}

        while len(ready_op_ids) > 0:
            logical_op_id = ready_op_ids.pop(0)
            logical_op = logical_plan.operators[logical_op_id]
            logical_source_op_ids = sorted(source_op_ids[logical_op_id])
            topological_order.append(logical_op_id)

            if isinstance(logical_op, BaseScan):
                source_cost_estimates = OperatorCostEstimates(
                    cardinality=len(logical_op.datasource),
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    quality=1.0,
                )
                right_source_cost_estimates = None
            elif isinstance(logical_op, ContextScan):
                source_cost_estimates = OperatorCostEstimates(
                    cardinality=1.0,
                    time_per_record=0.0,
                    cost_per_record=0.0,
                    quality=1.0,
                )
                right_source_cost_estimates = None
            elif isinstance(logical_op, LogicalJoinOp):
                if len(logical_source_op_ids) != 2:
                    raise ValueError(
                        f"Join logical op {logical_op_id} expected 2 source operators, "
                        f"found {len(logical_source_op_ids)}"
                    )
                source_cost_estimates = op_cost_estimates[logical_source_op_ids[0]]
                right_source_cost_estimates = op_cost_estimates[
                    logical_source_op_ids[1]
                ]
            else:
                if len(logical_source_op_ids) != 1:
                    raise ValueError(
                        f"Logical op {logical_op_id} expected 1 source operator, "
                        f"found {len(logical_source_op_ids)}"
                    )
                source_cost_estimates = op_cost_estimates[logical_source_op_ids[0]]
                right_source_cost_estimates = None

            op_source_cost_estimates[logical_op_id] = source_cost_estimates
            op_right_source_cost_estimates[logical_op_id] = right_source_cost_estimates
            cluster = self.clustering_strategy.build_cluster(
                logical_op,
                op_candidates[logical_op_id],
                source_op_cost_estimates=source_cost_estimates,
                right_source_op_cost_estimates=right_source_cost_estimates,
            )
            if cluster.cost_estimates is None:
                raise ValueError(
                    f"Cluster for logical op {logical_op_id} has no cost estimates"
                )

            op_clusters[logical_op_id] = cluster
            op_cost_estimates[logical_op_id] = cluster.cost_estimates

            for downstream_op_id in sorted(logical_plan.edges.get(logical_op_id, [])):
                remaining_source_counts[downstream_op_id] -= 1
                if remaining_source_counts[downstream_op_id] == 0:
                    ready_op_ids.append(downstream_op_id)
                    ready_op_ids.sort()

        if len(op_clusters) != len(logical_plan.operators):
            raise ValueError(
                "Unable to build physical operator clusters for cyclic logical plan"
            )

        return (
            op_clusters,
            source_op_ids,
            topological_order,
            op_source_cost_estimates,
            op_right_source_cost_estimates,
        )

    def build_sampling_progress_plan(
        self,
        logical_plan: LogicalPlan,
        op_clusters: dict[str, PhysicalOperatorCluster],
        source_op_ids: dict[str, list[str]],
        topological_order: list[str],
    ) -> SentinelPlan:
        """Build a display-only sentinel plan for cluster sampling progress."""
        progress_plans = {}
        for logical_op_id in topological_order:
            subplans = [
                progress_plans[source_op_id]
                for source_op_id in sorted(source_op_ids[logical_op_id])
            ]
            progress_plans[logical_op_id] = SentinelPlan(
                op_clusters[logical_op_id].physical_ops,
                subplans=subplans,
            )

        return progress_plans[logical_plan.root_op_id]

    def sample_physical_operator_clusters(
        self,
        logical_plan: LogicalPlan,
        op_clusters: dict[str, PhysicalOperatorCluster],
        source_op_ids: dict[str, list[str]],
        topological_order: list[str],
        op_source_cost_estimates: dict[str, OperatorCostEstimates],
        op_right_source_cost_estimates: dict[str, OperatorCostEstimates | None],
        physical_operator_selector: PhysicalOperatorSelector,
        validator: Validator | None = None,
        optimization_stats: SentinelPlanStats | None = None,
        progress_manager: ProgressManager | None = None,
        progress_logical_op_ids: dict[str, str] | None = None,
    ) -> None:
        """Sample each logical operator cluster for the configured budget.

        This method drives the exploration/exploitation selector over every
        logical operator cluster in source-to-sink order. Each selected physical
        operator is executed on an actual sampled input record and the observed
        runtime/cost statistics are recorded back into the selected cluster path.
        """
        sampled_records_by_logical_op_id: dict[str, list[DataRecord]] = {}
        progress_total = self.optimizer_config.sample_budget * len(topological_order)

        for logical_op_id in topological_order:
            logical_op = logical_plan.operators[logical_op_id]
            cluster = op_clusters[logical_op_id]
            logical_source_op_ids = sorted(source_op_ids[logical_op_id])
            progress_logical_op_id = (
                progress_logical_op_ids.get(logical_op_id)
                if progress_logical_op_ids is not None
                else None
            )

            if isinstance(logical_op, BaseScan):
                input_payloads = {
                    record_idx: record_idx
                    for record_idx in range(len(logical_op.datasource))
                }
            elif isinstance(logical_op, ContextScan):
                input_payloads = {f"{logical_op_id}:context": None}
            elif isinstance(logical_op, LogicalJoinOp):
                left_records = sampled_records_by_logical_op_id.get(
                    logical_source_op_ids[0], []
                )
                right_records = sampled_records_by_logical_op_id.get(
                    logical_source_op_ids[1], []
                )
                input_payloads = {}
                pair_idx = 0
                for left_record in left_records:
                    for right_record in right_records:
                        input_payloads[
                            f"{left_record._id}:{right_record._id}:{pair_idx}"
                        ] = (left_record, right_record)
                        pair_idx += 1
            elif any(isinstance(op, AggregateOp) for op in cluster.physical_ops):
                source_records = sampled_records_by_logical_op_id.get(
                    logical_source_op_ids[0], []
                )
                input_payloads = (
                    {f"{logical_op_id}:aggregate": source_records}
                    if len(source_records) > 0
                    else {}
                )
            elif len(logical_source_op_ids) == 1:
                source_records = sampled_records_by_logical_op_id.get(
                    logical_source_op_ids[0], []
                )
                input_payloads = {
                    f"{record._id}:{record_idx}": record
                    for record_idx, record in enumerate(source_records)
                }
            else:
                input_payloads = {}

            sampled_records_by_logical_op_id[logical_op_id] = []
            if len(input_payloads) == 0:
                if progress_manager is not None and progress_logical_op_id is not None:
                    self.update_sampling_progress_total(
                        progress_manager,
                        progress_logical_op_id,
                        0,
                        progress_total,
                    )
                    progress_total -= self.optimizer_config.sample_budget
                continue

            input_record_ids = list(input_payloads)
            sampling_round_idx = 0
            samples_drawn = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while sampling_round_idx < self.optimizer_config.sample_budget:
                    selections = []
                    while (
                        sampling_round_idx < self.optimizer_config.sample_budget
                        and len(selections) < self.max_workers
                    ):
                        try:
                            selection = physical_operator_selector.select_next_sample(
                                cluster,
                                input_record_ids,
                                sampling_round_idx,
                            )
                        except ValueError:
                            sampling_round_idx = self.optimizer_config.sample_budget
                            break

                        physical_operator_selector.executed_physical_op_ids_by_record[
                            selection.record_id
                        ].add(selection.physical_op.get_full_op_id())
                        selections.append((sampling_round_idx, selection))
                        sampling_round_idx += 1

                    if len(selections) == 0:
                        break

                    futures = {
                        executor.submit(
                            self.execute_and_score_physical_operator_sample,
                            logical_op,
                            selection.physical_op,
                            input_payloads[selection.record_id],
                            validator,
                        ): (round_idx, selection)
                        for round_idx, selection in selections
                    }
                    sample_results = []
                    for future in as_completed(futures):
                        round_idx, selection = futures[future]
                        record_set, input_count, elapsed_time, validation_gen_stats = (
                            future.result()
                        )
                        sample_results.append(
                            (
                                round_idx,
                                selection,
                                record_set,
                                input_count,
                                elapsed_time,
                                validation_gen_stats,
                            )
                        )

                    for _, selection, record_set, input_count, elapsed_time, validation_gen_stats in sorted(
                        sample_results,
                        key=lambda result: result[0],
                    ):
                        if input_count == 0:
                            continue

                        if optimization_stats is not None:
                            optimization_stats.add_record_op_stats(
                                logical_op_id,
                                record_set.record_op_stats,
                            )
                            optimization_stats.add_validation_gen_stats(
                                logical_op_id,
                                validation_gen_stats,
                            )

                        source_cost_estimates = op_source_cost_estimates[logical_op_id]
                        right_source_cost_estimates = op_right_source_cost_estimates[
                            logical_op_id
                        ]
                        if len(logical_source_op_ids) > 0:
                            source_cluster_cost_estimates = op_clusters[
                                logical_source_op_ids[0]
                            ].cost_estimates
                            if source_cluster_cost_estimates is not None:
                                source_cost_estimates = source_cluster_cost_estimates
                        if len(logical_source_op_ids) > 1:
                            right_source_cluster_cost_estimates = op_clusters[
                                logical_source_op_ids[1]
                            ].cost_estimates
                            if right_source_cluster_cost_estimates is not None:
                                right_source_cost_estimates = (
                                    right_source_cluster_cost_estimates
                                )

                        physical_op_id = selection.physical_op.get_full_op_id()
                        naive_cost_estimates = (
                            selection.cluster_path[-1].physical_op_cost_estimates[
                                physical_op_id
                            ]
                        )
                        estimate_cardinality_from_sample = isinstance(
                            selection.physical_op,
                            (
                                ContextScanOp,
                                ConvertOp,
                                FilterOp,
                                PhysicalJoinOp,
                                ScanPhysicalOp,
                            ),
                        )
                        observed_cost_estimates = self.estimate_sample_cost(
                            record_set,
                            input_count,
                            source_cost_estimates,
                            naive_cost_estimates,
                            elapsed_time,
                            right_source_cost_estimates=right_source_cost_estimates,
                            estimate_cardinality_from_sample=estimate_cardinality_from_sample,
                        )
                        physical_operator_selector.record_execution(
                            selection,
                            observed_cost_estimates,
                        )
                        passed_records = [
                            record
                            for record in record_set.data_records
                            if record._passed_operator
                        ]
                        sampled_records_by_logical_op_id[logical_op_id].extend(
                            passed_records
                        )

                        if (
                            progress_manager is not None
                            and progress_logical_op_id is not None
                        ):
                            progress_cost = sum(
                                stats.cost_per_record
                                for stats in record_set.record_op_stats
                            )
                            progress_cost += validation_gen_stats.cost_per_record
                            progress_manager.incr(
                                progress_logical_op_id,
                                1,
                                display_text=(
                                    f"{selection.physical_op.op_name()} "
                                    f"({len(passed_records)} outputs)"
                                ),
                                total_cost=progress_cost,
                            )
                        samples_drawn += 1

            if samples_drawn < self.optimizer_config.sample_budget:
                if progress_manager is not None and progress_logical_op_id is not None:
                    self.update_sampling_progress_total(
                        progress_manager,
                        progress_logical_op_id,
                        samples_drawn,
                        progress_total,
                    )
                    progress_total -= self.optimizer_config.sample_budget - samples_drawn

    def update_sampling_progress_total(
        self,
        progress_manager: ProgressManager,
        progress_logical_op_id: str,
        samples_drawn: int,
        progress_total: int,
    ) -> None:
        """Shrink progress totals when an operator exhausts samples early."""
        if not hasattr(progress_manager, "op_progress"):
            return

        task = progress_manager.unique_logical_op_id_to_task.get(progress_logical_op_id)
        if task is not None:
            progress_manager.op_progress.update(task, total=samples_drawn)

        adjusted_total = progress_total - (
            self.optimizer_config.sample_budget - samples_drawn
        )
        progress_manager.overall_progress.update(
            progress_manager.overall_task_id,
            total=max(adjusted_total, 0),
            refresh=True,
        )
        progress_manager.live_display.refresh()

    def execute_and_score_physical_operator_sample(
        self,
        logical_op,
        physical_op: PhysicalOperator,
        input_payload,
        validator: Validator | None,
    ) -> tuple[DataRecordSet, int, float, GenerationStats]:
        """Execute one optimizer sample and score it with the validator if present."""
        record_set, input_count, elapsed_time = self.execute_physical_operator_sample(
            logical_op,
            physical_op,
            input_payload,
        )
        validation_gen_stats = self.score_sample_quality(
            validator,
            physical_op,
            record_set,
        )
        return record_set, input_count, elapsed_time, validation_gen_stats

    def execute_physical_operator_sample(
        self,
        logical_op,
        physical_op: PhysicalOperator,
        input_payload,
    ) -> tuple[DataRecordSet, int, float]:
        """Execute a copied physical operator on one sampled optimizer input."""
        execution_op = physical_op.copy()
        if isinstance(execution_op, PhysicalJoinOp):
            execution_op._left_input_records = []
            execution_op._right_input_records = []
            execution_op._left_joined_record_ids = set()
            execution_op._right_joined_record_ids = set()
            execution_op.join_idx = 0
            execution_op.finished = False
        if isinstance(execution_op, BatchedOperator):
            execution_op.flushed = False
            execution_op._buffer = []
        if hasattr(execution_op, "_distinct_seen"):
            execution_op._distinct_seen = set()

        start_time = time.time()
        if isinstance(logical_op, BaseScan):
            record_set = execution_op(input_payload)
            input_count = 1
            record_set.input = input_payload
        elif isinstance(logical_op, ContextScan):
            record_set = execution_op()
            input_count = 1
            record_set.input = input_payload
        elif isinstance(execution_op, PhysicalJoinOp):
            left_record, right_record = input_payload
            record_set, input_count = execution_op([left_record], [right_record])
            record_set.input = ([left_record], [right_record])
        elif isinstance(execution_op, AggregateOp):
            input_records = (
                input_payload if isinstance(input_payload, list) else [input_payload]
            )
            record_set = execution_op(candidates=input_records)
            input_count = len(input_records)
            record_set.input = input_records
        else:
            record_set = execution_op(input_payload)
            input_count = 1
            if (
                isinstance(execution_op, BatchedOperator)
                and len(record_set) == 0
                and execution_op.has_pending_batch()
            ):
                record_set = execution_op.flush()
            if record_set.input is None:
                record_set.input = input_payload

        return record_set, input_count, time.time() - start_time

    def score_sample_quality(
        self,
        validator: Validator | None,
        physical_op: PhysicalOperator,
        record_set: DataRecordSet,
    ) -> GenerationStats:
        """Populate sampled record qualities using the provided validator."""
        if len(record_set.record_op_stats) == 0:
            return GenerationStats()

        if not isinstance(physical_op, (LLMConvert, LLMFilter, TopKOp, PhysicalJoinOp)):
            for record_op_stats in record_set.record_op_stats:
                record_op_stats.quality = 1.0
            return GenerationStats()

        if validator is None:
            return GenerationStats()

        if isinstance(physical_op, LLMConvert):
            if len(record_set.data_records) == 0:
                return GenerationStats()
            fields = physical_op.generated_fields
            input_record: DataRecord = record_set.input
            if physical_op.cardinality is Cardinality.ONE_TO_ONE:
                output = record_set.data_records[0].to_dict(project_cols=fields)
                output_str = record_set.data_records[0].to_json_str(
                    project_cols=fields,
                    bytes_to_str=True,
                    sorted=True,
                )
                full_hash = f"{hash(input_record)}{hash(output_str)}"
                score, validation_gen_stats, _ = validator._score_map(
                    physical_op,
                    fields,
                    input_record,
                    output,
                    full_hash,
                )
                record_set.record_op_stats[0].quality = score
                return validation_gen_stats
            else:
                output = [
                    data_record.to_dict(project_cols=fields)
                    for data_record in record_set.data_records
                ]
                output_strs = [
                    data_record.to_json_str(
                        project_cols=fields,
                        bytes_to_str=True,
                        sorted=True,
                    )
                    for data_record in record_set.data_records
                ]
                full_hash = f"{hash(input_record)}{hash(tuple(sorted(output_strs)))}"
                score, validation_gen_stats, _ = validator._score_flat_map(
                    physical_op,
                    fields,
                    input_record,
                    output,
                    full_hash,
                )
                for record_op_stats in record_set.record_op_stats:
                    record_op_stats.quality = score
                return validation_gen_stats

        if isinstance(physical_op, TopKOp):
            if len(record_set.data_records) == 0:
                return GenerationStats()
            fields = physical_op.generated_fields
            input_record: DataRecord = record_set.input
            output = record_set.data_records[0].to_dict(project_cols=fields)
            output_str = record_set.data_records[0].to_json_str(
                project_cols=fields,
                bytes_to_str=True,
                sorted=True,
            )
            full_hash = f"{hash(input_record)}{hash(output_str)}"
            score, validation_gen_stats, _ = validator._score_topk(
                physical_op,
                fields,
                input_record,
                output,
                full_hash,
            )
            record_set.record_op_stats[0].quality = score
            return validation_gen_stats

        if isinstance(physical_op, LLMFilter):
            validation_gen_stats = GenerationStats()
            scoring_op = physical_op
            if isinstance(physical_op, BatchedOperator):
                op_params = physical_op.get_op_params()
                op_params.pop("batch_size", None)
                scoring_op = LLMFilter(**op_params)

            filter_str = scoring_op.filter_obj.filter_condition
            input_records = (
                record_set.input
                if isinstance(record_set.input, list)
                else [record_set.input]
            )
            for input_record, data_record, record_op_stats in zip(
                input_records,
                record_set.data_records,
                record_set.record_op_stats,
                strict=True,
            ):
                output = data_record._passed_operator
                full_hash = f"{filter_str}{hash(input_record)}"
                score, sample_validation_gen_stats, _ = validator._score_filter(
                    scoring_op,
                    filter_str,
                    input_record,
                    output,
                    full_hash,
                )
                validation_gen_stats += sample_validation_gen_stats
                record_op_stats.quality = score
            return validation_gen_stats

        if isinstance(physical_op, PhysicalJoinOp):
            validation_gen_stats = GenerationStats()
            condition = physical_op.condition
            left_records, right_records = record_set.input
            record_idx = 0
            for left_record in left_records:
                for right_record in right_records:
                    data_record = record_set.data_records[record_idx]
                    output = data_record._passed_operator
                    full_hash = (
                        f"{condition}{hash(left_record)}{hash(right_record)}"
                    )
                    score, sample_validation_gen_stats, _ = validator._score_join(
                        physical_op,
                        condition,
                        left_record,
                        right_record,
                        output,
                        full_hash,
                    )
                    validation_gen_stats += sample_validation_gen_stats
                    record_set.record_op_stats[record_idx].quality = score
                    record_idx += 1
            return validation_gen_stats

        return GenerationStats()

    def estimate_sample_cost(
        self,
        record_set: DataRecordSet,
        input_count: int,
        source_cost_estimates: OperatorCostEstimates,
        naive_cost_estimates: OperatorCostEstimates,
        elapsed_time: float,
        right_source_cost_estimates: OperatorCostEstimates | None = None,
        estimate_cardinality_from_sample: bool = True,
    ) -> OperatorCostEstimates:
        """Convert sampled execution stats into operator cost estimates."""
        input_count = max(input_count, 1)
        record_op_stats = record_set.record_op_stats
        total_time = sum(stats.time_per_record for stats in record_op_stats)
        total_cost = sum(stats.cost_per_record for stats in record_op_stats)
        output_count = sum(
            record._passed_operator for record in record_set.data_records
        )

        input_cardinality = source_cost_estimates.cardinality
        if right_source_cost_estimates is not None:
            input_cardinality *= right_source_cost_estimates.cardinality

        cardinality = naive_cost_estimates.cardinality
        if estimate_cardinality_from_sample:
            cardinality = (output_count / input_count) * input_cardinality

        observed_qualities = [
            stats.quality for stats in record_op_stats if stats.quality is not None
        ]
        quality = (
            sum(observed_qualities) / len(observed_qualities)
            if len(observed_qualities) > 0
            else naive_cost_estimates.quality
        )

        return OperatorCostEstimates(
            cardinality=cardinality,
            time_per_record=(
                total_time / input_count
                if len(record_op_stats) > 0
                else elapsed_time / input_count
            ),
            cost_per_record=(
                total_cost / input_count
                if len(record_op_stats) > 0
                else naive_cost_estimates.cost_per_record
            ),
            quality=quality,
        )

    def select_physical_operators(
        self,
        op_clusters: dict[str, PhysicalOperatorCluster],
        topological_order: list[str],
        physical_operator_selector: PhysicalOperatorSelector,
    ) -> tuple[
        dict[str, PhysicalOperator],
        dict[str, OperatorCostEstimates],
    ]:
        """Select one concrete physical operator for every logical operator."""
        selected_physical_ops = {}
        selected_op_cost_estimates = {}
        for logical_op_id in topological_order:
            physical_op, cost_estimates = (
                physical_operator_selector.select_best_physical_operator(
                    op_clusters[logical_op_id]
                )
            )
            selected_physical_ops[logical_op_id] = physical_op
            selected_op_cost_estimates[logical_op_id] = cost_estimates

        return selected_physical_ops, selected_op_cost_estimates

    def build_physical_plan(
        self,
        logical_plan: LogicalPlan,
        source_op_ids: dict[str, list[str]],
        topological_order: list[str],
        op_source_cost_estimates: dict[str, OperatorCostEstimates],
        op_right_source_cost_estimates: dict[str, OperatorCostEstimates | None],
        selected_physical_ops: dict[str, PhysicalOperator],
        selected_op_cost_estimates: dict[str, OperatorCostEstimates],
    ) -> PhysicalPlan:
        """Assemble a ``PhysicalPlan`` from the selected physical operators."""
        physical_plans = {}
        for logical_op_id in topological_order:
            source_ids = sorted(source_op_ids[logical_op_id])
            subplans = [physical_plans[source_op_id] for source_op_id in source_ids]
            physical_op = selected_physical_ops[logical_op_id]
            op_cost_estimates = selected_op_cost_estimates[logical_op_id]
            if len(source_ids) == 0:
                source_cost_estimates = op_source_cost_estimates[logical_op_id]
                right_source_cost_estimates = op_right_source_cost_estimates[
                    logical_op_id
                ]
            elif len(source_ids) == 1:
                source_cost_estimates = selected_op_cost_estimates[source_ids[0]]
                right_source_cost_estimates = None
            elif len(source_ids) == 2:
                source_cost_estimates = selected_op_cost_estimates[source_ids[0]]
                right_source_cost_estimates = selected_op_cost_estimates[source_ids[1]]
            else:
                raise ValueError(
                    f"Logical op {logical_op_id} has {len(subplans)} source plans"
                )

            input_cardinality = source_cost_estimates.cardinality
            if right_source_cost_estimates is not None:
                input_cardinality *= right_source_cost_estimates.cardinality
            op_plan_cost = PlanCost(
                cost=op_cost_estimates.cost_per_record * input_cardinality,
                time=op_cost_estimates.time_per_record * input_cardinality,
                quality=op_cost_estimates.quality,
                op_estimates=op_cost_estimates,
            )

            if len(subplans) == 0:
                plan_cost = op_plan_cost
            elif len(subplans) == 1:
                plan_cost = subplans[0].plan_cost + op_plan_cost
            elif len(subplans) == 2:
                optimizer_execution_strategy = self.optimizer_config.execution_strategy
                is_parallel_execution = (
                    optimizer_execution_strategy.is_fully_parallel()
                    if hasattr(optimizer_execution_strategy, "is_fully_parallel")
                    else str(optimizer_execution_strategy).lower() == "parallel"
                )
                execution_strategy = (
                    "parallel" if is_parallel_execution else "sequential"
                )
                plan_cost = op_plan_cost.join_add(
                    subplans[0].plan_cost,
                    subplans[1].plan_cost,
                    execution_strategy=execution_strategy,
                )

            physical_plans[logical_op_id] = PhysicalPlan(
                physical_op,
                subplans=subplans,
                plan_cost=plan_cost,
            )

        return physical_plans[logical_plan.root_op_id]

    def optimize(
        self,
        logical_plan: LogicalPlan,
        validator: Validator | None = None,
        optimization_stats: SentinelPlanStats | None = None,
    ) -> PhysicalPlan:
        """Optimize a logical plan and return a physical plan."""
        physical_operator_selector = PhysicalOperatorSelector(
            policy=self.policy,
            total_sampling_rounds=self.optimizer_config.sample_budget,
            seed=self.optimizer_config.seed,
        )

        (
            op_clusters,
            source_op_ids,
            topological_order,
            op_source_cost_estimates,
            op_right_source_cost_estimates,
        ) = self.build_physical_operator_clusters(logical_plan)

        if optimization_stats is not None:
            optimization_stats.operator_stats = {}
            for logical_op_id in topological_order:
                optimization_stats.operator_stats[logical_op_id] = {}
                for physical_op in op_clusters[logical_op_id].physical_ops:
                    optimization_stats.operator_stats[logical_op_id][
                        physical_op.get_full_op_id()
                    ] = OperatorStats(
                        full_op_id=physical_op.get_full_op_id(),
                        op_name=physical_op.op_name(),
                        source_unique_logical_op_ids=source_op_ids[logical_op_id],
                        plan_id=optimization_stats.plan_id,
                        op_details={
                            key: str(value)
                            for key, value in physical_op.get_id_params().items()
                        },
                    )

        progress_manager = None
        progress_logical_op_ids = {}
        if self.progress:
            progress_plan = self.build_sampling_progress_plan(
                logical_plan,
                op_clusters,
                source_op_ids,
                topological_order,
            )
            progress_manager = create_progress_manager(
                progress_plan,
                sample_budget=(
                    self.optimizer_config.sample_budget * len(topological_order)
                ),
                sample_cost_budget=None,
                progress=self.progress,
            )
            for topo_idx, (progress_logical_op_id, _) in enumerate(progress_plan):
                logical_op_id = topological_order[topo_idx]
                unique_progress_logical_op_id = (
                    f"{topo_idx}-{progress_logical_op_id}"
                )
                progress_logical_op_ids[logical_op_id] = unique_progress_logical_op_id
                task = progress_manager.unique_logical_op_id_to_task.get(
                    unique_progress_logical_op_id
                )
                if task is not None:
                    progress_manager.op_progress.update(
                        task,
                        total=self.optimizer_config.sample_budget,
                    )

        if progress_manager is not None:
            progress_manager.start()
        try:
            self.sample_physical_operator_clusters(
                logical_plan,
                op_clusters,
                source_op_ids,
                topological_order,
                op_source_cost_estimates,
                op_right_source_cost_estimates,
                physical_operator_selector,
                validator=validator,
                optimization_stats=optimization_stats,
                progress_manager=progress_manager,
                progress_logical_op_ids=progress_logical_op_ids,
            )
        finally:
            if progress_manager is not None:
                progress_manager.finish()

        selected_physical_ops, selected_op_cost_estimates = (
            self.select_physical_operators(
                op_clusters,
                topological_order,
                physical_operator_selector,
            )
        )

        optimized_plan = self.build_physical_plan(
            logical_plan,
            source_op_ids,
            topological_order,
            op_source_cost_estimates,
            op_right_source_cost_estimates,
            selected_physical_ops,
            selected_op_cost_estimates,
        )

        return optimized_plan
