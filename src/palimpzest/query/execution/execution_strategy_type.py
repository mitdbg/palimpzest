from enum import Enum

from palimpzest.query.execution.all_sample_execution_strategy import AllSamplingExecutionStrategy
from palimpzest.query.execution.mab_execution_strategy import MABExecutionStrategy
from palimpzest.query.execution.parallel_execution_strategy import (
    ParallelExecutionStrategy,
    SequentialParallelExecutionStrategy,
)
from palimpzest.query.execution.random_sampling_execution_strategy import RandomSamplingExecutionStrategy
from palimpzest.query.execution.single_threaded_execution_strategy import (
    PipelinedSingleThreadExecutionStrategy,
    SequentialSingleThreadExecutionStrategy,
)


class ExecutionStrategyType(Enum):
    """Available execution strategy types"""
    SEQUENTIAL = SequentialSingleThreadExecutionStrategy
    PIPELINED = PipelinedSingleThreadExecutionStrategy
    PARALLEL = ParallelExecutionStrategy
    SEQUENTIAL_PARALLEL = SequentialParallelExecutionStrategy

class SentinelExecutionStrategyType(Enum):
    MAB = MABExecutionStrategy
    RANDOM = RandomSamplingExecutionStrategy
    ALL = AllSamplingExecutionStrategy
