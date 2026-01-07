import logging
import os
from enum import Enum

from dotenv import load_dotenv

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.query.execution.execution_strategy import ExecutionStrategy, SentinelExecutionStrategy
from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType, SentinelExecutionStrategyType
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor import QueryProcessor
from palimpzest.utils.model_helpers import get_models
from palimpzest.validator.validator import Validator

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
            "execution_strategy": ExecutionStrategyType,
            "sentinel_execution_strategy": SentinelExecutionStrategyType,
            "optimizer_strategy": OptimizationStrategyType,
        }
        for strategy in ["execution_strategy", "sentinel_execution_strategy", "optimizer_strategy"]:
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
    def _config_validation_and_normalization(cls, config: QueryProcessorConfig, train_dataset: dict[str, Dataset] | None, validator : Validator | None):
        if config.policy is None:
            raise ValueError("Policy is required for optimizer")
        
        # only one of progress or verbose can be set; we will default to progress=True
        if config.progress and config.verbose:
            print("WARNING: Both `progress` and `verbose` are set to True, but only one can be True at a time; defaulting to `progress=True`")
            config.verbose = False

        # if the user provides a training dataset, but no validator, create a default validator
        if train_dataset is not None and validator is None:
            validator = Validator()
            logger.info("No validator provided; using default Validator")

        # boolean flag for whether we're performing optimization or not
        optimization = validator is not None

        # handle "auto" default for sentinel execution strategies
        if config.sentinel_execution_strategy == "auto":
            config.sentinel_execution_strategy = "mab" if optimization else None

        # convert the config values for processing, execution, and optimization strategies to enums
        config = cls._normalize_strategies(config)

        # get available models
        available_models = getattr(config, 'available_models', [])
        if available_models is None or len(available_models) == 0:
            available_models = get_models(use_vertex=config.use_vertex, gemini_credentials_path=config.gemini_credentials_path, api_base=config.api_base)

        # remove any models specified in the config
        remove_models = getattr(config, 'remove_models', [])
        if remove_models is not None and len(remove_models) > 0:
            available_models = [model for model in available_models if model not in remove_models]
            logger.info(f"Removed models from available models based on config: {remove_models}")

        # set the final set of available models in the config
        config.available_models = available_models

        if len(config.available_models) == 0:
            raise ValueError("No available models found.")

        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        together_key = os.getenv("TOGETHER_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        for model in config.available_models:
            if model.is_openai_model() and not openai_key:
                raise ValueError("OPENAI_API_KEY must be set to use OpenAI models.")
            if model.is_anthropic_model() and not anthropic_key:
                raise ValueError("ANTHROPIC_API_KEY must be set to use Anthropic models.")
            if model.is_together_model() and not together_key:
                raise ValueError("TOGETHER_API_KEY must be set to use Together models.")
            if model.is_google_ai_studio_model() and not (gemini_key or google_key or config.gemini_credentials_path):
                raise ValueError("GEMINI_API_KEY, GOOGLE_API_KEY, or gemini_credentials path must be set to use Google Gemini models.")
            if model.is_vllm_model() and config.api_base is None:
                raise ValueError("api_base must be set to use vLLM models.")

        return config, validator

    @classmethod
    def _create_optimizer(cls, config: QueryProcessorConfig) -> Optimizer:
        return Optimizer(cost_model=SampleBasedCostModel(), **config.to_dict())

    @classmethod
    def _create_execution_strategy(cls, dataset: Dataset, config: QueryProcessorConfig) -> ExecutionStrategy:
        """
        Creates an execution strategy based on the configuration.
        """
        # for parallel execution, set the batch size if there's a limit in the query
        limit = dataset.get_limit()
        if limit is not None and config.execution_strategy == ExecutionStrategyType.PARALLEL:
            if config.batch_size is None:
                config.batch_size = limit
                logger.info(f"Setting batch size to query limit: {limit}")
            elif config.batch_size > limit:
                config.batch_size = limit
                logger.info(f"Setting batch size to query limit: {limit} since it was larger than the limit")

        # create the execution strategy
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
        dataset: Dataset,
        config: QueryProcessorConfig | None = None,
        train_dataset: dict[str, Dataset] | None = None,
        validator: Validator | None = None,
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

        # make a copy of the config to avoid modifying the original
        config = config.copy()

        # apply any additional keyword arguments to the config and validate its contents
        config, validator = cls._config_validation_and_normalization(config, train_dataset, validator)

        # update the dataset's types if we're not enforcing types
        if not config.enforce_types:
            dataset.relax_types()
            if train_dataset is not None:
                for _, ds in train_dataset.items():
                    ds.relax_types()

        # create the optimizer, execution strateg(ies), and processor
        optimizer = cls._create_optimizer(config)
        config.execution_strategy = cls._create_execution_strategy(dataset, config)
        config.sentinel_execution_strategy = cls._create_sentinel_execution_strategy(config)
        processor = QueryProcessor(dataset, optimizer, train_dataset=train_dataset, validator=validator, **config.to_dict())

        return processor

    @classmethod
    def create_and_run_processor(
        cls,
        dataset: Dataset,
        config: QueryProcessorConfig | None = None,
        train_dataset: dict[str, Dataset] | None = None,
        validator: Validator | None = None,
    ) -> DataRecordCollection:
        load_dotenv(override=True)
        logger.info(f"Creating processor for dataset: {dataset}")
        processor = cls.create_processor(dataset, config, train_dataset, validator)
        logger.info(f"Created processor: {processor}")

        return processor.execute()
