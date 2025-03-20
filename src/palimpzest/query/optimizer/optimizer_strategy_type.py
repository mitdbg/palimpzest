from enum import Enum

from palimpzest.query.optimizer.optimizer_strategy import (
    GreedyStrategy,
    NoOptimizationStrategy,
    ParetoStrategy,
    SentinelStrategy,
)


class OptimizationStrategyType(Enum):
    """
    OptimizationStrategyType determines which (set of) plan(s) the Optimizer
    will return to the Execution layer.
    """
    GREEDY = GreedyStrategy
    PARETO = ParetoStrategy
    SENTINEL = SentinelStrategy
    NONE = NoOptimizationStrategy

    def no_transformation(self) -> bool:
        """
        Return True if this optimization strategy does not transform the logical plan.
        """
        return self in [OptimizationStrategyType.SENTINEL, OptimizationStrategyType.NONE]

    def is_pareto(self) -> bool:
        """
        Return True if this optimization strategy uses Pareto optimization.
        """
        return self == OptimizationStrategyType.PARETO

    def is_not_pareto(self) -> bool:
        """
        Return True if this optimization strategy does not use Pareto optimization.
        """
        return not self.is_pareto()
