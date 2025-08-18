from enum import Enum

from palimpzest.query.execution.all_sample_execution_strategy import AllSamplingExecutionStrategy
from palimpzest.query.execution.mab_execution_strategy import MABExecutionStrategy
from palimpzest.query.execution.parallel_execution_strategy import ParallelExecutionStrategy
from palimpzest.query.execution.single_threaded_execution_strategy import (
    PipelinedSingleThreadExecutionStrategy,
    SequentialSingleThreadExecutionStrategy,
)


class ExecutionStrategyType(Enum):
    """Available execution strategy types"""
    SEQUENTIAL = SequentialSingleThreadExecutionStrategy
    PIPELINED = PipelinedSingleThreadExecutionStrategy
    PARALLEL = ParallelExecutionStrategy

    def is_fully_parallel(self) -> bool:
        """Check if the execution strategy executes operators in parallel."""
        return self == ExecutionStrategyType.PARALLEL

class SentinelExecutionStrategyType(Enum):
    MAB = MABExecutionStrategy
    ALL = AllSamplingExecutionStrategy
