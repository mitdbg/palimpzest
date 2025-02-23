import logging
from enum import Enum

from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.execution.execution_strategy import ExecutionStrategyType
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy import OptimizationStrategyType
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.mab_sentinel_processor import (
    MABSentinelPipelinedParallelProcessor,
    MABSentinelSequentialSingleThreadProcessor,
)
from palimpzest.query.processor.nosentinel_processor import (
    NoSentinelPipelinedParallelProcessor,
    NoSentinelPipelinedSingleThreadProcessor,
    NoSentinelSequentialSingleThreadProcessor,
)
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.query.processor.random_sampling_sentinel_processor import (
    RandomSamplingSentinelPipelinedParallelProcessor,
    RandomSamplingSentinelSequentialSingleThreadProcessor,
)
from palimpzest.query.processor.streaming_processor import StreamingQueryProcessor
from palimpzest.sets import Dataset, Set
from palimpzest.utils.model_helpers import get_models

logger = logging.getLogger(__name__)

class ProcessingStrategyType(Enum):
    """How to generate and optimize query plans"""
    MAB_SENTINEL = "mab_sentinel"
    NO_SENTINEL = "nosentinel"
    RANDOM_SAMPLING = "random_sampling"
    STREAMING = "streaming"
    AUTO = "auto"

def convert_to_enum(enum_type: type[Enum], value: str) -> Enum:
    if value == "pipelined":
        value = "pipelined_single_thread"
    value = value.upper().replace('-', '_')
    try:
        return enum_type[value]
    except KeyError as e:
        raise ValueError(f"Unsupported {enum_type.__name__}: {value}") from e


class QueryProcessorFactory:
    PROCESSOR_MAPPING = {
        (ProcessingStrategyType.NO_SENTINEL, ExecutionStrategyType.SEQUENTIAL): 
            NoSentinelSequentialSingleThreadProcessor,
        (ProcessingStrategyType.NO_SENTINEL, ExecutionStrategyType.PIPELINED_SINGLE_THREAD): 
            NoSentinelPipelinedSingleThreadProcessor,
        (ProcessingStrategyType.NO_SENTINEL, ExecutionStrategyType.PIPELINED_PARALLEL): 
            NoSentinelPipelinedParallelProcessor,
        (ProcessingStrategyType.MAB_SENTINEL, ExecutionStrategyType.SEQUENTIAL):
            MABSentinelSequentialSingleThreadProcessor,
        (ProcessingStrategyType.MAB_SENTINEL, ExecutionStrategyType.PIPELINED_PARALLEL):
            MABSentinelPipelinedParallelProcessor,
        (ProcessingStrategyType.STREAMING, ExecutionStrategyType.SEQUENTIAL):
            StreamingQueryProcessor,
        (ProcessingStrategyType.STREAMING, ExecutionStrategyType.PIPELINED_PARALLEL):
            StreamingQueryProcessor,
        (ProcessingStrategyType.RANDOM_SAMPLING, ExecutionStrategyType.SEQUENTIAL):
            RandomSamplingSentinelSequentialSingleThreadProcessor,
        (ProcessingStrategyType.RANDOM_SAMPLING, ExecutionStrategyType.PIPELINED_PARALLEL):
            RandomSamplingSentinelPipelinedParallelProcessor,
    }

    @classmethod
    def create_processor(
        cls,
        dataset: Set,
        config: QueryProcessorConfig | None = None,
        **kwargs
    ) -> QueryProcessor:
        """
        Creates a QueryProcessor with specified processing and execution strategies.

        Args:
            dataset: The dataset to process
            config: The user-provided QueryProcessorConfig; if it is None, the default config will be used
            kwargs: Additional keyword arguments to pass to the QueryProcessorConfig
        """
        if config is None:
            config = QueryProcessorConfig()

        # apply any additional keyword arguments to the config
        config.update(**kwargs)

        config = cls._config_validation_and_normalization(config)
        processing_strategy, execution_strategy, optimizer_strategy = cls._normalize_strategies(config)
        optimizer = cls._create_optimizer(optimizer_strategy, config)

        processor_key = (processing_strategy, execution_strategy)
        processor_cls = cls.PROCESSOR_MAPPING.get(processor_key)

        if processor_cls is None:
            raise ValueError(f"Unsupported combination of processing strategy {processing_strategy} "
                        f"and execution strategy {execution_strategy}")

        return processor_cls(dataset=dataset, optimizer=optimizer, config=config, **kwargs)

    @classmethod
    def create_and_run_processor(cls, dataset: Dataset, config: QueryProcessorConfig | None = None, **kwargs) -> DataRecordCollection:
        # TODO(Jun): Consider to use cache here.
        logger.info(f"Creating processor for dataset: {dataset}")
        processor = cls.create_processor(dataset=dataset, config=config, **kwargs)
        logger.info(f"Created processor: {processor}")
        return processor.execute()

    #TODO(Jun): The all avaliable plans could be generated earlier and outside Optimizer.
    @classmethod
    def _create_optimizer(cls, optimizer_strategy: OptimizationStrategyType, config: QueryProcessorConfig) -> Optimizer:
        available_models = getattr(config, 'available_models', []) or get_models(include_vision=True)
        
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")
        
        return Optimizer(
            policy=config.policy,
            cost_model=CostModel(),
            cache=config.cache,
            verbose=config.verbose,
            available_models=available_models,
            allow_bonded_query=config.allow_bonded_query,
            allow_code_synth=config.allow_code_synth,
            allow_token_reduction=config.allow_token_reduction,
            allow_rag_reduction=config.allow_rag_reduction,
            allow_mixtures=config.allow_mixtures,
            allow_critic=config.allow_critic,
            optimization_strategy_type=optimizer_strategy,
            use_final_op_quality=config.use_final_op_quality
        )

    @classmethod
    def _normalize_strategies(cls, config: QueryProcessorConfig):
        processing_strategy, execution_strategy, optimizer_strategy = config.processing_strategy, config.execution_strategy, config.optimizer_strategy
        
        if isinstance(processing_strategy, str):
            try:
                processing_strategy = convert_to_enum(ProcessingStrategyType, processing_strategy)
            except ValueError as e:
                raise ValueError(f"""Unsupported processing strategy: {processing_strategy}.
                                    The supported strategies are: {ProcessingStrategyType.__members__.keys()}""") from e
        if isinstance(execution_strategy, str):
            try:
                execution_strategy = convert_to_enum(ExecutionStrategyType, execution_strategy)
            except ValueError as e:
                raise ValueError(f"""Unsupported execution strategy: {execution_strategy}. 
                                    The supported strategies are: {ExecutionStrategyType.__members__.keys()}""") from e
        if isinstance(optimizer_strategy, str):
            try:
                optimizer_strategy = convert_to_enum(OptimizationStrategyType, optimizer_strategy)
            except ValueError as e:
                raise ValueError(f"""Unsupported optimizer strategy: {optimizer_strategy}. 
                                    The supported strategies are: {OptimizationStrategyType.__members__.keys()}""") from e
        return processing_strategy, execution_strategy, optimizer_strategy

    @classmethod
    def _config_validation_and_normalization(cls, config: QueryProcessorConfig):
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")

        if config.cache:
            raise ValueError("cache=True is not supported yet")

        if config.val_datasource is None and config.processing_strategy in [ProcessingStrategyType.MAB_SENTINEL, ProcessingStrategyType.RANDOM_SAMPLING]:
            raise ValueError("val_datasource is required for MAB_SENTINEL and RANDOM_SAMPLING processing strategies")

        available_models = getattr(config, 'available_models', [])
        if available_models is None or len(available_models) == 0:
            available_models = get_models(include_vision=True)
        config.available_models = available_models

        return config
