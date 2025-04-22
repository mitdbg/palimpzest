import logging
from enum import Enum

from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.execution.execution_strategy import ExecutionStrategy, SentinelExecutionStrategy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType, SentinelExecutionStrategyType
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.processing_strategy_type import ProcessingStrategyType
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.sets import Dataset, Set
from palimpzest.utils.model_helpers import get_models

logger = logging.getLogger(__name__)


class QueryProcessorFactory:

    @classmethod
    def _convert_to_enum(cls, enum_type: type[Enum], value: str) -> Enum:
        value = value.upper().replace('-', '_')
        try:
            return enum_type[value]
        except KeyError as e:
            raise ValueError(f"Unsupported {enum_type.__name__}: {value}") from e

    @classmethod
    def _normalize_strategies(cls, config: QueryProcessorConfig):
        """
        Convert the string representation of each strategy into its Enum equivalent and throw
        an exception if the conversion fails.
        """
        strategy_types = {
            "processing_strategy": ProcessingStrategyType,
            "execution_strategy": ExecutionStrategyType,
            "sentinel_execution_strategy": SentinelExecutionStrategyType,
            "optimizer_strategy": OptimizationStrategyType,
        }
        for strategy in ["processing_strategy", "execution_strategy", "sentinel_execution_strategy", "optimizer_strategy"]:
            strategy_str = getattr(config, strategy)
            strategy_type = strategy_types[strategy]
            strategy_enum = None
            if strategy_str is not None:
                try:
                    strategy_enum = cls._convert_to_enum(strategy_type, strategy_str)
                except ValueError as e:
                    raise ValueError(f"""Unsupported {strategy}: {strategy_str}.
                                        The supported strategies are: {strategy_type.__members__.keys()}""") from e
            setattr(config, strategy, strategy_enum)
            logger.debug(f"Normalized {strategy}: {strategy_enum}")

        return config

    @classmethod
    def _config_validation_and_normalization(cls, config: QueryProcessorConfig):
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")

        if config.cache:
            raise ValueError("cache=True is not supported yet")
        
        # only one of progress or verbose can be set; we will default to progress=True
        if config.progress and config.verbose:
            print("WARNING: Both `progress` and `verbose` are set to True, but only one can be True at a time; defaulting to `progress=True`")
            config.verbose = False

        # handle "auto" defaults for processing and sentinel execution strategies
        if config.processing_strategy == "auto":
            config.processing_strategy = "no_sentinel" if config.val_datasource is None else "sentinel"

        if config.sentinel_execution_strategy == "auto":
            config.sentinel_execution_strategy = None if config.val_datasource is None else "mab"

        # convert the config values for processing, execution, and optimization strategies to enums
        config = cls._normalize_strategies(config)

        # check that processor uses a supported execution strategy
        if config.execution_strategy not in config.processing_strategy.valid_execution_strategies():
            raise ValueError(f"Unsupported `execution_strategy` {config.execution_strategy} for `processing_strategy` {config.processing_strategy}.")

        # check that validation data is provided for sentinel execution
        if config.val_datasource is None and config.processing_strategy.is_sentinel_strategy():
            raise ValueError("`val_datasource` is required for SENTINEL processing strategies")

        # check that sentinel execution is provided for sentinel processor
        if config.sentinel_execution_strategy is None and config.processing_strategy.is_sentinel_strategy():
            raise ValueError("`sentinel_execution_strategy` is required for SENTINEL processing strategies")

        # get available models
        available_models = getattr(config, 'available_models', [])
        if available_models is None or len(available_models) == 0:
            available_models = get_models(include_vision=True)
        config.available_models = available_models

        return config

    @classmethod
    def _create_optimizer(cls, config: QueryProcessorConfig) -> Optimizer:
        return Optimizer(cost_model=CostModel(), **config.to_dict())

    @classmethod
    def _create_execution_strategy(cls, config: QueryProcessorConfig) -> ExecutionStrategy:
        """
        Creates an execution strategy based on the configuration.
        """
        execution_strategy_cls = config.execution_strategy.value
        return execution_strategy_cls(**config.to_dict())

    @classmethod
    def _create_sentinel_execution_strategy(cls, config: QueryProcessorConfig) -> SentinelExecutionStrategy:
        """
        Creates an execution strategy based on the configuration.
        """
        if config.sentinel_execution_strategy is None:
            return None

        sentinel_execution_strategy_cls = config.sentinel_execution_strategy.value
        return sentinel_execution_strategy_cls(**config.to_dict())

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

        # apply any additional keyword arguments to the config and validate its contents
        config.update(**kwargs)
        config = cls._config_validation_and_normalization(config)

        # create the optimizer, execution strateg(ies), and processor
        optimizer = cls._create_optimizer(config)
        config.execution_strategy = cls._create_execution_strategy(config)
        config.sentinel_execution_strategy = cls._create_sentinel_execution_strategy(config)
        processor_cls = config.processing_strategy.value
        processor = processor_cls(dataset, optimizer, **config.to_dict())

        return processor

    @classmethod
    def create_and_run_processor(cls, dataset: Dataset, config: QueryProcessorConfig | None = None, **kwargs) -> DataRecordCollection:
        # TODO(Jun): Consider to use cache here.
        logger.info(f"Creating processor for dataset: {dataset}")
        processor = cls.create_processor(dataset=dataset, config=config, **kwargs)
        logger.info(f"Created processor: {processor}")

        return processor.execute()
