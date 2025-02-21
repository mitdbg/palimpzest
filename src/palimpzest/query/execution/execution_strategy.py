import logging
import time
from abc import ABC, abstractmethod
from enum import Enum

from palimpzest.core.data.dataclasses import ExecutionStats, PlanStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.tools.logger import setup_logger

logger = setup_logger(__name__)


class ExecutionStrategyType(str, Enum):
    """Available execution strategy types"""
    SEQUENTIAL = "sequential"
    PIPELINED_SINGLE_THREAD = "pipelined"
    PIPELINED_PARALLEL = "pipelined_parallel"
    AUTO = "auto"


class ExecutionStrategy(ABC):
    """
    Base strategy for executing query plans.
    Defines how to execute a single plan.
    """
    def __init__(self, 
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 cache: bool = False,
                 verbose: bool = False):
        self.scan_start_idx = scan_start_idx
        self.cache = cache
        self.verbose = verbose
        self.max_workers = max_workers
        self.execution_stats = []
        logger.pz_logger.set_console_level(logging.DEBUG if verbose else logging.ERROR)
        logger.info(f"Initialized ExecutionStrategy {self.__class__.__name__}")
        logger.debug(f"ExecutionStrategy initialized with config: {self.__dict__}")

    @abstractmethod
    def execute_plan(
        self,
        plan: PhysicalPlan,
        num_samples: int | float = float("inf"),
        workers: int = 1
    ) -> tuple[list[DataRecord], PlanStats]:
        """Execute a single plan according to strategy"""
        pass


    # TODO(chjun): use _create_execution_stats for execution stats setup.
    ## aggregate plan stats
    # aggregate_plan_stats = self.aggregate_plan_stats(plan_stats)

    # # add sentinel records and plan stats (if captured) to plan execution data
    # execution_stats = ExecutionStats(
    #     execution_id=self.execution_id(),
    #     plan_stats=aggregate_plan_stats,
    #     total_execution_time=time.time() - execution_start_time,
    #     total_execution_cost=sum(
    #         list(map(lambda plan_stats: plan_stats.total_plan_cost, aggregate_plan_stats.values()))
    #     ),
    #     plan_strs={plan_id: plan_stats.plan_str for plan_id, plan_stats in aggregate_plan_stats.items()},
    # )
    def _create_execution_stats(
        self,
        plan_stats: list[PlanStats],
        start_time: float
    ) -> ExecutionStats:
        """Create execution statistics"""
        return ExecutionStats(
            execution_id=f"exec_{int(start_time)}",
            plan_stats={ps.plan_id: ps for ps in plan_stats},
            total_execution_time=time.time() - start_time,
            total_execution_cost=sum(ps.total_cost for ps in plan_stats)
        )
