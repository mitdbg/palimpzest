import logging
from abc import abstractmethod

from palimpzest.core.data.dataclasses import PlanStats
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import ExecutionStrategy, SentinelExecutionStrategy
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.sets import Dataset
from palimpzest.utils.hash_helpers import hash_for_id
from palimpzest.utils.model_helpers import get_models

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes queries through the complete pipeline:
    1. Optimization phase: Plan generation and selection
    2. Execution phase: Plan execution and result collection
    3. Result phase: Statistics gathering and result formatting
    """
    def __init__(
        self,
        dataset: Dataset,
        optimizer: Optimizer,
        execution_strategy: ExecutionStrategy,
        sentinel_execution_strategy: SentinelExecutionStrategy | None,
        num_samples: int | None = None,
        val_datasource: DataReader | None = None,
        scan_start_idx: int = 0,
        cache: bool = False,
        verbose: bool = False,
        progress: bool = True,
        max_workers: int | None = None,
        num_workers_per_plan: int = 1,
        min_plans: int = 1,
        policy: Policy | None = None,
        available_models: list[str] | None = None,
        **kwargs,
    ):
        """
        Initialize QueryProcessor with optional custom components.
        
        Args:
            dataset: Dataset to process
            TODO
        """
        self.dataset = dataset
        self.optimizer = optimizer
        self.execution_strategy = execution_strategy
        self.sentinel_execution_strategy = sentinel_execution_strategy

        self.num_samples = num_samples
        self.val_datasource = val_datasource
        self.scan_start_idx = scan_start_idx
        self.cache = cache
        self.verbose = verbose
        self.progress = progress
        self.max_workers = max_workers
        self.num_workers_per_plan = num_workers_per_plan
        self.min_plans = min_plans

        self.policy = policy

        self.available_models = available_models
        if self.available_models is None or len(self.available_models) == 0:
            self.available_models = get_models(include_vision=True)

        if self.verbose:
            print("Available models: ", self.available_models)

        logger.info(f"Initialized QueryProcessor {self.__class__.__name__}")
        logger.debug(f"QueryProcessor initialized with config: {self.__dict__}")

    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"

        return hash_for_id(id_str)

    def aggregate_plan_stats(self, plan_stats: list[PlanStats]) -> dict[str, PlanStats]:
        """
        Aggregate a list of plan stats into a dictionary mapping plan_id --> cumulative plan stats.

        NOTE: we make the assumption that the same plan cannot be run more than once in parallel,
        i.e. each plan stats object for an individual plan comes from two different (sequential)
        periods in time. Thus, PlanStats' total_plan_time(s) can be summed.
        """
        agg_plan_stats = {}
        for ps in plan_stats:
            if ps.plan_id in agg_plan_stats:
                agg_plan_stats[ps.plan_id] += ps
            else:
                agg_plan_stats[ps.plan_id] = ps

        return agg_plan_stats
    
    def _execute_best_plan(
        self,
        dataset: Dataset,
        policy: Policy,
        optimizer: Optimizer,
    ) -> tuple[list[DataRecord], list[PlanStats]]:
        # get the optimal plan according to the optimizer
        plans = optimizer.optimize(dataset, policy)
        final_plan = plans[0]

        # execute the plan
        # TODO: for some reason this is not picking up change to self.max_workers from ParallelPlanExecutor.__init__()
        records, plan_stats = self.execution_strategy.execute_plan(plan=final_plan)

        # return the output records and plan stats
        return records, [plan_stats]

    # TODO: consider to support dry_run.
    @abstractmethod
    def execute(self) -> DataRecordCollection:
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
