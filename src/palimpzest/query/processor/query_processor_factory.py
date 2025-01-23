from enum import Enum
from typing import Type

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
    NoSentinelPipelinedSinglelProcessor,
    NoSentinelSequentialSingleThreadProcessor,
)
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.query.processor.random_sampling_sentinel_processor import (
    RandomSamplingSentinelPipelinedProcessor,
    RandomSamplingSentinelSequentialSingleThreadProcessor,
)
from palimpzest.query.processor.streaming_processor import StreamingQueryProcessor
from palimpzest.sets import Dataset
from palimpzest.utils.model_helpers import get_models


class ProcessingStrategyType(Enum):
    """How to generate and optimize query plans"""
    MAB_SENTINEL = "mab_sentinel"
    NO_SENTINEL = "nosentinel"
    RANDOM_SAMPLING = "random_sampling"
    STREAMING = "streaming"
    AUTO = "auto"

def convert_to_enum(enum_type: Type[Enum], value: str) -> Enum:
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
            lambda ds, opt, cfg: NoSentinelSequentialSingleThreadProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.NO_SENTINEL, ExecutionStrategyType.PIPELINED_SINGLE_THREAD): 
            lambda ds, opt, cfg: NoSentinelPipelinedSinglelProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.NO_SENTINEL, ExecutionStrategyType.PIPELINED_PARALLEL): 
            lambda ds, opt, cfg: NoSentinelPipelinedParallelProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.MAB_SENTINEL, ExecutionStrategyType.SEQUENTIAL):
            lambda ds, opt, cfg: MABSentinelSequentialSingleThreadProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.MAB_SENTINEL, ExecutionStrategyType.PIPELINED_PARALLEL):
            lambda ds, opt, cfg: MABSentinelPipelinedParallelProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.STREAMING, ExecutionStrategyType.SEQUENTIAL):
            lambda ds, opt, cfg: StreamingQueryProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.STREAMING, ExecutionStrategyType.PIPELINED_PARALLEL):
            lambda ds, opt, cfg: StreamingQueryProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.RANDOM_SAMPLING, ExecutionStrategyType.SEQUENTIAL):
            lambda ds, opt, cfg: RandomSamplingSentinelSequentialSingleThreadProcessor(datasource=ds, optimizer=opt, config=cfg),
        (ProcessingStrategyType.RANDOM_SAMPLING, ExecutionStrategyType.PIPELINED_PARALLEL):
            lambda ds, opt, cfg: RandomSamplingSentinelPipelinedProcessor(datasource=ds, optimizer=opt, config=cfg),
    }


    @staticmethod
    def create_processor(
        datasource: Dataset,
        processing_strategy: str | ProcessingStrategyType = ProcessingStrategyType.NO_SENTINEL,
        execution_strategy: str | ExecutionStrategyType = ExecutionStrategyType.SEQUENTIAL,
        optimizer_strategy: str | OptimizationStrategyType = OptimizationStrategyType.PARETO,
        config: QueryProcessorConfig | None = None,
    ) -> QueryProcessor:
        """
        Creates a QueryProcessor with specified processing and execution strategies.

        Args:
            datasource: The data source to process
            processing_strategy: How to generate/optimize query plans and execute them
            execution_strategy: How to execute the plans
            optimizer_strategy: How to find the optimal plan
            config: Additional configuration parameters:
        """
        if config is None:
            config = QueryProcessorConfig()

        # Normalize enum values
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
            
        # intialize an optimizer with the strategy
        available_models = getattr(config, 'available_models', [])
        if available_models is None or len(available_models) == 0:
            available_models = get_models(include_vision=True)
        
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")
        config.available_models = available_models

        optimizer = QueryProcessorFactory._create_optimizer(optimizer_strategy, config)

        # Get the appropriate processor based on strategy combination
        processor_key = (processing_strategy, execution_strategy)
        processor_factory = QueryProcessorFactory.PROCESSOR_MAPPING.get(processor_key)
        
        if processor_factory is None:
            raise ValueError(f"Unsupported combination of processing strategy {processing_strategy} "
                           f"and execution strategy {execution_strategy}")

        return processor_factory(datasource, optimizer, config)


    #TODO(Jun): The all avaliable plans could be generated earlier and outside Optimizer.
    @staticmethod
    def _create_optimizer(optimizer_strategy: OptimizationStrategyType, config: QueryProcessorConfig) -> Optimizer:
        available_models = getattr(config, 'available_models', []) or get_models(include_vision=True)
        
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")
        
        return Optimizer(
            policy=config.policy,
            cost_model=CostModel(),
            no_cache=config.nocache,
            verbose=config.verbose,
            available_models=available_models,
            allow_bonded_query=config.allow_bonded_query,
            allow_conventional_query=config.allow_conventional_query,
            allow_code_synth=config.allow_code_synth,
            allow_token_reduction=config.allow_token_reduction,
            optimization_strategy_type=optimizer_strategy,
            use_final_op_quality=config.use_final_op_quality
        )
