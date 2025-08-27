import logging

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.models import ExecutionStats, PlanStats
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import ExecutionStrategy, SentinelExecutionStrategy
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.hash_helpers import hash_for_id
from palimpzest.validator.validator import Validator

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
        train_dataset: dict[str, Dataset] | None = None,
        validator: Validator | None = None,
        scan_start_idx: int = 0,
        verbose: bool = False,
        progress: bool = True,
        max_workers: int | None = None,
        policy: Policy | None = None,
        available_models: list[str] | None = None,
        **kwargs,  # needed in order to provide compatibility with QueryProcessorConfig
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
        self.train_dataset = train_dataset
        self.validator = validator
        self.scan_start_idx = scan_start_idx
        self.verbose = verbose
        self.progress = progress
        self.max_workers = max_workers
        self.policy = policy
        self.available_models = available_models

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

    def _create_sentinel_plan(self, train_dataset: dict[str, Dataset] | None) -> SentinelPlan:
        """
        Generates and returns a SentinelPlan for the given dataset.
        """
        # create a new optimizer and update its strategy to SENTINEL
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_strategy(OptimizationStrategyType.SENTINEL)

        # create copy of dataset, but change its root Dataset(s) to the validation Dataset(s)
        dataset = self.dataset.copy()
        if train_dataset is not None:
            dataset._set_root_datasets(train_dataset)
            dataset._generate_unique_logical_op_ids()

        # get the sentinel plan for the given dataset
        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]

        return sentinel_plan

    def _execute_best_plan(self, dataset: Dataset, optimizer: Optimizer) -> tuple[list[DataRecord], list[PlanStats]]:
        # get the optimal plan according to the optimizer
        plans = optimizer.optimize(dataset)
        final_plan = plans[0]

        # execute the plan
        records, plan_stats = self.execution_strategy.execute_plan(plan=final_plan)

        # return the output records and plan stats
        return records, [plan_stats]

    def execute(self) -> DataRecordCollection:
        logger.info(f"Executing {self.__class__.__name__}")

        # create execution stats
        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        # if the user provides a validator, we perform optimization
        if self.validator is not None:
            # create sentinel plan
            sentinel_plan = self._create_sentinel_plan(self.train_dataset)

            # generate sample execution data
            if self.train_dataset is not None:
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, self.train_dataset, self.validator)

            else:
                train_dataset = self.dataset._get_root_datasets()
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, train_dataset, self.validator)

            # update the execution stats to account for the work done in optimization
            execution_stats.add_plan_stats(sentinel_plan_stats)
            execution_stats.finish_optimization()

            # (re-)initialize the optimizer
            self.optimizer = self.optimizer.deepcopy_clean()

            # construct the CostModel with any sample execution data we've gathered
            cost_model = SampleBasedCostModel(sentinel_plan_stats, self.verbose)
            self.optimizer.update_cost_model(cost_model)

        # execute plan(s) according to the optimization strategy
        records, plan_stats = self._execute_best_plan(self.dataset, self.optimizer)

        # update the execution stats to account for the work to execute the final plan
        execution_stats.add_plan_stats(plan_stats)
        execution_stats.finish()

        # construct and return the DataRecordCollection
        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info(f"Done executing {self.__class__.__name__}")

        return result
