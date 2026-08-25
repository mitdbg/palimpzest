from __future__ import annotations

from itertools import combinations, product

from palimpzest.constants import AggFunc, Model, PromptStrategy
from palimpzest.core.lib.schemas import (
    AUDIO_FIELD_TYPES,
    AUDIO_LIST_FIELD_TYPES,
    IMAGE_FIELD_TYPES,
    IMAGE_LIST_FIELD_TYPES,
)
from palimpzest.query.operators.aggregate import (
    ApplyGroupByOp,
    AverageAggregateOp,
    CountAggregateOp,
    MaxAggregateOp,
    MinAggregateOp,
    SemanticAggregate,
    SumAggregateOp,
)
from palimpzest.query.operators.batched import BatchedFilter
from palimpzest.query.operators.compute import SmolAgentsCompute
from palimpzest.query.operators.convert import LLMConvertBonded, NonLLMConvert
from palimpzest.query.operators.critique_and_refine import (
    CritiqueAndRefineConvert,
    CritiqueAndRefineFilter,
)
from palimpzest.query.operators.distinct import DistinctOp
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.image_filter import RescaledImageFilter
from palimpzest.query.operators.join import (
    EmbeddingJoin,
    NestedLoopsJoin,
    RelationalJoin,
)
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.logical import (
    Aggregate,
    BaseScan,
    ComputeOperator,
    ContextScan,
    ConvertScan,
    Distinct,
    FilteredScan,
    GroupByAggregate,
    JoinOp,
    LimitScan,
    LogicalOperator,
    Project,
    SearchOperator,
    TopKScan,
)
from palimpzest.query.operators.mixture_of_agents import (
    MixtureOfAgentsConvert,
    MixtureOfAgentsFilter,
)
from palimpzest.query.operators.physical import PhysicalOperator
from palimpzest.query.operators.project import ProjectOp
from palimpzest.query.operators.rag import RAGConvert, RAGFilter
from palimpzest.query.operators.scan import ContextScanOp, MarshalAndScanDataOp
from palimpzest.query.operators.search import SmolAgentsSearch
from palimpzest.query.operators.split import SplitConvert, SplitFilter
from palimpzest.query.operators.topk import TopKOp
from palimpzest.query.optimizer_config import OptimizerConfig
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.cluster.logical_optimizer import LogicalPlan
from palimpzest.utils.model_helpers import use_reasoning_prompt

RAG_NUM_CHUNKS_PER_FIELD = [1, 2, 4]
RAG_CHUNK_SIZES = [1000, 2000, 4000]
SPLIT_NUM_CHUNKS = [2, 4, 6]
SPLIT_MIN_SIZE_TO_CHUNK = [1000, 4000]
BATCH_SIZES = [10, 50, 100]
TOPK_K_BUDGETS = [1, 3, 5, 10, 15, 20, 25]
RESCALE_FACTORS = [2, 3, 4]
MOA_NUM_PROPOSER_MODELS = [1, 2, 3]
MOA_TEMPERATURES = [0.0, 0.4, 0.8]
DEFAULT_NUM_JOIN_SAMPLES = 10

NON_SEMANTIC_LOGICAL_OP_TO_PHYSICAL_OP = {
    BaseScan: MarshalAndScanDataOp,
    ContextScan: ContextScanOp,
    ConvertScan: NonLLMConvert,
    Distinct: DistinctOp,
    FilteredScan: NonLLMFilter,
    GroupByAggregate: ApplyGroupByOp,
    JoinOp: RelationalJoin,
    LimitScan: LimitScanOp,
    Project: ProjectOp,
}


def _short_field_name(field_name: str) -> str:
    return field_name.split(".")[-1]


def _get_input_fields(logical_op: LogicalOperator) -> dict:
    if logical_op.input_schema is None:
        return {}
    return logical_op.input_schema.model_fields


def _get_depends_on_field_names(logical_op: LogicalOperator) -> set[str]:
    input_fields = _get_input_fields(logical_op)
    if len(input_fields) == 0:
        return set()

    if len(logical_op.depends_on) == 0:
        return set(input_fields)

    return {_short_field_name(field_name) for field_name in logical_op.depends_on}


def _dependent_field_annotations(logical_op: LogicalOperator) -> list[object]:
    depends_on_field_names = _get_depends_on_field_names(logical_op)
    return [
        field.annotation
        for field_name, field in _get_input_fields(logical_op).items()
        if _short_field_name(field_name) in depends_on_field_names
    ]


def _get_fields_with_annotations(
    logical_op: LogicalOperator, field_types: list[object]
) -> set[str]:
    depends_on_field_names = _get_depends_on_field_names(logical_op)
    return {
        _short_field_name(field_name)
        for field_name, field in _get_input_fields(logical_op).items()
        if field.annotation in field_types
        and _short_field_name(field_name) in depends_on_field_names
    }


def _is_image_only_operation(logical_op: LogicalOperator) -> bool:
    field_annotations = _dependent_field_annotations(logical_op)
    return len(field_annotations) > 0 and all(
        field_type in IMAGE_FIELD_TYPES for field_type in field_annotations
    )


def _is_image_operation(logical_op: LogicalOperator) -> bool:
    return any(
        field_type in IMAGE_FIELD_TYPES
        for field_type in _dependent_field_annotations(logical_op)
    )


def _is_audio_only_operation(logical_op: LogicalOperator) -> bool:
    field_annotations = _dependent_field_annotations(logical_op)
    return len(field_annotations) > 0 and all(
        field_type in AUDIO_FIELD_TYPES for field_type in field_annotations
    )


def _is_audio_operation(logical_op: LogicalOperator) -> bool:
    return any(
        field_type in AUDIO_FIELD_TYPES
        for field_type in _dependent_field_annotations(logical_op)
    )


def _is_text_only_operation(logical_op: LogicalOperator) -> bool:
    field_annotations = _dependent_field_annotations(logical_op)
    return len(field_annotations) > 0 and all(
        field_type not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
        for field_type in field_annotations
    )


def _is_text_operation(logical_op: LogicalOperator) -> bool:
    return any(
        field_type not in IMAGE_FIELD_TYPES + AUDIO_FIELD_TYPES
        for field_type in _dependent_field_annotations(logical_op)
    )


def _model_matches_input(model: Model, logical_op: LogicalOperator) -> bool:
    num_image_fields = len(_get_fields_with_annotations(logical_op, IMAGE_FIELD_TYPES))
    num_list_image_field = len(
        _get_fields_with_annotations(logical_op, IMAGE_LIST_FIELD_TYPES)
    )

    num_audio_fields = len(_get_fields_with_annotations(logical_op, AUDIO_FIELD_TYPES))
    num_list_audio_fields = len(
        _get_fields_with_annotations(logical_op, AUDIO_LIST_FIELD_TYPES)
    )

    if model.is_embedding_model():
        return False

    if (
        model.is_llama_model()
        and model.is_vision_model()
        and (num_image_fields > 1 or num_list_image_field > 0)
    ):
        return False

    if (
        model.is_provider_vertex_ai()
        and model.is_audio_model()
        and (num_audio_fields > 1 or num_list_audio_fields > 0)
    ):
        return False

    if _is_text_only_operation(logical_op) and model.is_text_model():
        return True

    if _is_image_only_operation(logical_op) and model.is_vision_model():
        return True

    if _is_audio_only_operation(logical_op) and model.is_audio_model():
        return True

    if (
        _is_image_operation(logical_op)
        and _is_text_operation(logical_op)
        and model.is_text_image_multimodal_model()
    ):
        return True

    if (
        _is_audio_operation(logical_op)
        and _is_text_operation(logical_op)
        and model.is_text_audio_multimodal_model()
    ):
        return True

    return False


def _embedding_model_matches_input(model: Model, logical_op: LogicalOperator) -> bool:
    if (
        _is_text_operation(logical_op)
        and _is_image_operation(logical_op)
        and model.is_text_image_multimodal_embedding_model()
    ):
        return True

    is_text_embedding_model = (
        model.is_embedding_model()
        and not model.is_text_image_multimodal_embedding_model()
    )
    return _is_text_only_operation(logical_op) and is_text_embedding_model


def _get_fixed_op_kwargs(
    logical_op: LogicalOperator,
    optimizer_config: OptimizerConfig,
) -> dict:
    op_kwargs = logical_op.get_logical_op_params()
    op_kwargs.update(
        {
            "depends_on": sorted(_get_depends_on_field_names(logical_op)),
            "verbose": optimizer_config.verbose,
            "logical_op_id": logical_op.get_logical_op_id(),
            "unique_logical_op_id": logical_op.unique_logical_op_id,
            "logical_op_name": logical_op.logical_op_name(),
        }
    )
    return op_kwargs


def _instantiate_physical_ops(
    logical_op: LogicalOperator,
    physical_op_class: type[PhysicalOperator],
    optimizer_config: OptimizerConfig,
    variable_op_kwargs: list[dict] | dict | None = None,
) -> list[PhysicalOperator]:
    fixed_op_kwargs = _get_fixed_op_kwargs(logical_op, optimizer_config)
    if variable_op_kwargs is None:
        variable_op_kwargs = [{}]
    elif isinstance(variable_op_kwargs, dict):
        variable_op_kwargs = [variable_op_kwargs]

    physical_ops = []
    seen_op_ids = set()
    for var_op_kwargs in variable_op_kwargs:
        op = physical_op_class(**{**fixed_op_kwargs, **var_op_kwargs})
        full_op_id = op.get_full_op_id()
        if full_op_id in seen_op_ids:
            continue
        seen_op_ids.add(full_op_id)
        physical_ops.append(op)

    return physical_ops


def _filter_prompt_strategy(optimizer_config: OptimizerConfig) -> PromptStrategy:
    return (
        PromptStrategy.FILTER
        if use_reasoning_prompt(optimizer_config.reasoning_effort)
        else PromptStrategy.FILTER_NO_REASONING
    )


def _map_prompt_strategy(optimizer_config: OptimizerConfig) -> PromptStrategy:
    return (
        PromptStrategy.MAP
        if use_reasoning_prompt(optimizer_config.reasoning_effort)
        else PromptStrategy.MAP_NO_REASONING
    )


def _join_prompt_strategy(optimizer_config: OptimizerConfig) -> PromptStrategy:
    return (
        PromptStrategy.JOIN
        if use_reasoning_prompt(optimizer_config.reasoning_effort)
        else PromptStrategy.JOIN_NO_REASONING
    )


def _agg_prompt_strategy(optimizer_config: OptimizerConfig) -> PromptStrategy:
    return (
        PromptStrategy.AGG
        if use_reasoning_prompt(optimizer_config.reasoning_effort)
        else PromptStrategy.AGG_NO_REASONING
    )


def _semantic_filter_ops(
    logical_op: FilteredScan,
    optimizer_config: OptimizerConfig,
) -> list[PhysicalOperator]:
    models = [
        model
        for model in optimizer_config.available_models
        if _model_matches_input(model, logical_op)
    ]
    reasoning_effort = optimizer_config.reasoning_effort
    prompt_strategy = _filter_prompt_strategy(optimizer_config)

    physical_ops = _instantiate_physical_ops(
        logical_op,
        LLMFilter,
        optimizer_config,
        [
            {
                "model": model,
                "prompt_strategy": prompt_strategy,
                "reasoning_effort": reasoning_effort,
            }
            for model in models
        ],
    )

    physical_ops.extend(
        _instantiate_physical_ops(
            logical_op,
            BatchedFilter,
            optimizer_config,
            [
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "reasoning_effort": reasoning_effort,
                    "batch_size": batch_size,
                }
                for model in models
                for batch_size in BATCH_SIZES
            ],
        )
    )

    if _is_image_operation(logical_op):
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                RescaledImageFilter,
                optimizer_config,
                [
                    {
                        "model": model,
                        "prompt_strategy": prompt_strategy,
                        "reasoning_effort": reasoning_effort,
                        "rescale_factor": rescale_factor,
                    }
                    for model in models
                    for rescale_factor in RESCALE_FACTORS
                ],
            )
        )

    if optimizer_config.allow_rag_reduction and _is_text_only_operation(logical_op):
        embedding_models = [
            model
            for model in optimizer_config.available_models
            if _embedding_model_matches_input(model, logical_op)
        ]
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                RAGFilter,
                optimizer_config,
                [
                    {
                        "model": model,
                        "embedding_model": embedding_model,
                        "prompt_strategy": prompt_strategy,
                        "num_chunks_per_field": num_chunks_per_field,
                        "chunk_size": chunk_size,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model, embedding_model in product(models, embedding_models)
                    for num_chunks_per_field in RAG_NUM_CHUNKS_PER_FIELD
                    for chunk_size in RAG_CHUNK_SIZES
                ],
            )
        )

    if optimizer_config.allow_split_merge and _is_text_only_operation(logical_op):
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                SplitFilter,
                optimizer_config,
                [
                    {
                        "model": model,
                        "min_size_to_chunk": min_size_to_chunk,
                        "num_chunks": num_chunks,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model in models
                    for min_size_to_chunk in SPLIT_MIN_SIZE_TO_CHUNK
                    for num_chunks in SPLIT_NUM_CHUNKS
                ],
            )
        )

    if optimizer_config.allow_mixtures:
        aggregator_models = [
            model
            for model in optimizer_config.available_models
            if model.is_text_model()
        ]
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                MixtureOfAgentsFilter,
                optimizer_config,
                [
                    {
                        "proposer_models": list(proposer_models),
                        "temperatures": [temperature] * len(proposer_models),
                        "aggregator_model": aggregator_model,
                        "reasoning_effort": reasoning_effort,
                    }
                    for num_proposers in MOA_NUM_PROPOSER_MODELS
                    for temperature in MOA_TEMPERATURES
                    for proposer_models in combinations(models, num_proposers)
                    for aggregator_model in aggregator_models
                ],
            )
        )

    if optimizer_config.allow_critic:
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                CritiqueAndRefineFilter,
                optimizer_config,
                [
                    {
                        "model": model,
                        "critic_model": critic_model,
                        "refine_model": refine_model,
                        "prompt_strategy": prompt_strategy,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model in models
                    for critic_model in models
                    for refine_model in models
                ],
            )
        )

    return physical_ops


def _semantic_convert_ops(
    logical_op: ConvertScan,
    optimizer_config: OptimizerConfig,
) -> list[PhysicalOperator]:
    models = [
        model
        for model in optimizer_config.available_models
        if _model_matches_input(model, logical_op)
    ]
    reasoning_effort = optimizer_config.reasoning_effort
    prompt_strategy = _map_prompt_strategy(optimizer_config)
    physical_ops = []

    if optimizer_config.allow_bonded_query:
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                LLMConvertBonded,
                optimizer_config,
                [
                    {
                        "model": model,
                        "prompt_strategy": prompt_strategy,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model in models
                ],
            )
        )

    if optimizer_config.allow_rag_reduction and _is_text_only_operation(logical_op):
        embedding_models = [
            model
            for model in optimizer_config.available_models
            if _embedding_model_matches_input(model, logical_op)
        ]
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                RAGConvert,
                optimizer_config,
                [
                    {
                        "model": model,
                        "embedding_model": embedding_model,
                        "prompt_strategy": prompt_strategy,
                        "num_chunks_per_field": num_chunks_per_field,
                        "chunk_size": chunk_size,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model, embedding_model in product(models, embedding_models)
                    for num_chunks_per_field in RAG_NUM_CHUNKS_PER_FIELD
                    for chunk_size in RAG_CHUNK_SIZES
                ],
            )
        )

    if optimizer_config.allow_split_merge and _is_text_only_operation(logical_op):
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                SplitConvert,
                optimizer_config,
                [
                    {
                        "model": model,
                        "min_size_to_chunk": min_size_to_chunk,
                        "num_chunks": num_chunks,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model in models
                    for min_size_to_chunk in SPLIT_MIN_SIZE_TO_CHUNK
                    for num_chunks in SPLIT_NUM_CHUNKS
                ],
            )
        )

    if optimizer_config.allow_mixtures:
        aggregator_models = [
            model
            for model in optimizer_config.available_models
            if model.is_text_model()
        ]
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                MixtureOfAgentsConvert,
                optimizer_config,
                [
                    {
                        "proposer_models": list(proposer_models),
                        "temperatures": [temperature] * len(proposer_models),
                        "aggregator_model": aggregator_model,
                        "reasoning_effort": reasoning_effort,
                    }
                    for num_proposers in MOA_NUM_PROPOSER_MODELS
                    for temperature in MOA_TEMPERATURES
                    for proposer_models in combinations(models, num_proposers)
                    for aggregator_model in aggregator_models
                ],
            )
        )

    if optimizer_config.allow_critic:
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                CritiqueAndRefineConvert,
                optimizer_config,
                [
                    {
                        "model": model,
                        "critic_model": critic_model,
                        "refine_model": refine_model,
                        "prompt_strategy": prompt_strategy,
                        "reasoning_effort": reasoning_effort,
                    }
                    for model in models
                    for critic_model in models
                    for refine_model in models
                ],
            )
        )

    return physical_ops


def _join_ops(
    logical_op: JoinOp,
    optimizer_config: OptimizerConfig,
) -> list[PhysicalOperator]:
    models = [
        model
        for model in optimizer_config.available_models
        if _model_matches_input(model, logical_op)
    ]
    reasoning_effort = optimizer_config.reasoning_effort
    prompt_strategy = _join_prompt_strategy(optimizer_config)
    join_parallelism = optimizer_config.join_parallelism
    retain_inputs = (
        optimizer_config.optimizer_strategy != OptimizationStrategyType.SENTINEL
    )
    variable_op_kwargs = [
        {
            "model": model,
            "prompt_strategy": prompt_strategy,
            "join_parallelism": join_parallelism,
            "reasoning_effort": reasoning_effort,
            "retain_inputs": retain_inputs,
        }
        for model in models
    ]
    physical_ops = _instantiate_physical_ops(
        logical_op,
        NestedLoopsJoin,
        optimizer_config,
        variable_op_kwargs,
    )

    if not _is_audio_operation(logical_op):
        embedding_models = [
            model
            for model in optimizer_config.available_models
            if _embedding_model_matches_input(model, logical_op)
        ]
        physical_ops.extend(
            _instantiate_physical_ops(
                logical_op,
                EmbeddingJoin,
                optimizer_config,
                [
                    {
                        "model": model,
                        "embedding_model": embedding_model,
                        "prompt_strategy": prompt_strategy,
                        "join_parallelism": join_parallelism,
                        "reasoning_effort": reasoning_effort,
                        "retain_inputs": retain_inputs,
                        "num_samples": DEFAULT_NUM_JOIN_SAMPLES,
                    }
                    for model, embedding_model in product(models, embedding_models)
                ],
            )
        )

    return physical_ops


def _aggregate_ops(
    logical_op: Aggregate,
    optimizer_config: OptimizerConfig,
) -> list[PhysicalOperator]:
    if logical_op.is_semantic:
        models = [
            model
            for model in optimizer_config.available_models
            if _model_matches_input(model, logical_op) and not model.is_llama_model()
        ]
        prompt_strategy = _agg_prompt_strategy(optimizer_config)
        reasoning_effort = optimizer_config.reasoning_effort
        return _instantiate_physical_ops(
            logical_op,
            SemanticAggregate,
            optimizer_config,
            [
                {
                    "model": model,
                    "prompt_strategy": prompt_strategy,
                    "reasoning_effort": reasoning_effort,
                }
                for model in models
            ],
        )

    aggregate_op_class = {
        AggFunc.COUNT: CountAggregateOp,
        AggFunc.AVERAGE: AverageAggregateOp,
        AggFunc.SUM: SumAggregateOp,
        AggFunc.MIN: MinAggregateOp,
        AggFunc.MAX: MaxAggregateOp,
    }.get(logical_op.agg_func)
    if aggregate_op_class is None:
        raise ValueError(f"Unsupported aggregate function: {logical_op.agg_func}")

    return _instantiate_physical_ops(logical_op, aggregate_op_class, optimizer_config)


def find_physical_candidates(
    op: LogicalOperator,
    optimizer_config: OptimizerConfig,
) -> list[PhysicalOperator]:
    """
    Instantiate all concrete physical operators that can implement a given logical
    operator in a Cluster logical plan.
    """
    if not op.is_semantic:
        op_class = NON_SEMANTIC_LOGICAL_OP_TO_PHYSICAL_OP.get(type(op))
        if op_class is None:
            raise NotImplementedError(f"No physical op exists for {op}")

        return _instantiate_physical_ops(
            op,
            op_class,
            optimizer_config,
        )

    else:
        if isinstance(op, FilteredScan):
            return _semantic_filter_ops(op, optimizer_config)

        elif isinstance(op, ConvertScan):
            return _semantic_convert_ops(op, optimizer_config)

        elif isinstance(op, TopKScan):
            ks = TOPK_K_BUDGETS if op.k == -1 else [op.k]
            return _instantiate_physical_ops(
                op,
                TopKOp,
                optimizer_config,
                [{"k": k} for k in ks],
            )

        elif isinstance(op, JoinOp):
            return _join_ops(op, optimizer_config)

        elif isinstance(op, Aggregate):
            return _aggregate_ops(op, optimizer_config)

        elif isinstance(op, ComputeOperator):
            return _instantiate_physical_ops(op, SmolAgentsCompute, optimizer_config)

        elif isinstance(op, SearchOperator):
            return _instantiate_physical_ops(op, SmolAgentsSearch, optimizer_config)

        else:
            raise NotImplementedError("No physical operator registered for", op)
