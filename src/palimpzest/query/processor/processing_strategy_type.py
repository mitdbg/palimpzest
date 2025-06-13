from enum import Enum

from palimpzest.query.execution.execution_strategy_type import ExecutionStrategyType
from palimpzest.query.processor.nosentinel_processor import NoSentinelQueryProcessor
from palimpzest.query.processor.sentinel_processor import SentinelQueryProcessor
from palimpzest.query.processor.streaming_processor import StreamingQueryProcessor


class ProcessingStrategyType(Enum):
    """How to generate and optimize query plans"""
    SENTINEL = SentinelQueryProcessor
    NO_SENTINEL = NoSentinelQueryProcessor
    STREAMING = StreamingQueryProcessor

    def valid_execution_strategies(self) -> list[ExecutionStrategyType]:
        """
        Returns a list of valid execution strategies for the given processing strategy.
        """
        if self == ProcessingStrategyType.SENTINEL or self == ProcessingStrategyType.NO_SENTINEL:
            return [ExecutionStrategyType.SEQUENTIAL, ExecutionStrategyType.PIPELINED, ExecutionStrategyType.PARALLEL, ExecutionStrategyType.SEQUENTIAL_PARALLEL]
        elif self == ProcessingStrategyType.STREAMING:
            return [ExecutionStrategyType.PIPELINED, ExecutionStrategyType.PARALLEL, ExecutionStrategyType.SEQUENTIAL_PARALLEL]

    def is_sentinel_strategy(self) -> bool:
        """
        Returns True if the query processor associated with this strategy uses sentinel execution.
        """
        return self == ProcessingStrategyType.SENTINEL
