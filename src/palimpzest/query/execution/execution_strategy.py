import logging
import time
from abc import ABC, abstractmethod
from enum import Enum

from palimpzest.core.data.dataclasses import ExecutionStats, PlanStats
from palimpzest.core.elements.records import DataRecord
from palimpzest.query.operators.scan import ScanPhysicalOp
from palimpzest.query.optimizer.plan import PhysicalPlan

logger = logging.getLogger(__name__)

class ExecutionStrategyType(str, Enum):
    """Available execution strategy types"""
    SEQUENTIAL = "sequential"
    PIPELINED = "pipelined"
    PARALLEL = "parallel"
    AUTO = "auto"


class ExecutionStrategy(ABC):
    """
    Base strategy for executing query plans.
    Defines how to execute a single plan.
    """
    def __init__(self, 
                 scan_start_idx: int = 0, 
                 max_workers: int | None = None,
                 num_samples: int | None = None,
                 cache: bool = False,
                 verbose: bool = False,
                 progress: bool = True):
        self.scan_start_idx = scan_start_idx
        self.max_workers = max_workers
        self.num_samples = num_samples
        self.cache = cache
        self.verbose = verbose
        self.progress = progress

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

    def _add_records_to_cache(self, target_cache_id: str, records: list[DataRecord]) -> None:
        """Add each record (which isn't filtered) to the cache for the given target_cache_id."""
        if self.cache:
            for record in records:
                if getattr(record, "passed_operator", True):
                    # self.datadir.append_cache(target_cache_id, record)
                    pass

    def _close_cache(self, target_cache_ids: list[str]) -> None:
        """Close the cache for each of the given target_cache_ids"""
        if self.cache:
            for target_cache_id in target_cache_ids:  # noqa: B007
                # self.datadir.close_cache(target_cache_id)
                pass

    def _create_input_queues(self, plan: PhysicalPlan) -> dict[str, list]:
        """Initialize input queues for each operator in the plan."""
        input_queues = {}
        for op in plan.operators:
            inputs = []
            if isinstance(op, ScanPhysicalOp):
                scan_end_idx = (
                    len(op.datareader)
                    if self.num_samples is None
                    else min(self.scan_start_idx + self.num_samples, len(op.datareader))
                )
                inputs = [idx for idx in range(self.scan_start_idx, scan_end_idx)]
            input_queues[op.get_op_id()] = inputs

        return input_queues

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
