import logging
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

from palimpzest.core.data.dataclasses import PlanStats, RecordOpStats
from palimpzest.core.data.datareaders import DataReader
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.policy import Policy
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy import OptimizationStrategyType
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.sets import Dataset, Set
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
        optimizer: Optimizer = None,
        config: QueryProcessorConfig = None,
        *args,
        **kwargs,
    ):
        """
        Initialize QueryProcessor with optional custom components.
        
        Args:
            dataset: Dataset to process
            optimizer: Custom optimizer (optional)
            execution_engine: Custom execution engine (optional)
            config: Configuration dictionary for default components
        """
        assert config is not None, "QueryProcessorConfig is required for QueryProcessor"

        self.config = config or QueryProcessorConfig()
        self.dataset = dataset
        self.datareader = self._get_datareader(self.dataset)
        self.num_samples = self.config.num_samples
        self.val_datasource = self.config.val_datasource
        self.scan_start_idx = self.config.scan_start_idx
        self.cache = self.config.cache
        self.verbose = self.config.verbose
        self.max_workers = self.config.max_workers
        self.num_workers_per_plan = self.config.num_workers_per_plan
        self.min_plans = self.config.min_plans

        self.policy = self.config.policy

        self.available_models = self.config.available_models
        if self.available_models is None or len(self.available_models) == 0:
            self.available_models = get_models(include_vision=True)

        if self.verbose:
            print("Available models: ", self.available_models)

        # Initialize optimizer and execution engine
        # TODO: config currently has optimizer field which is string. 
        # In this case, we only use the initialized optimizer. Later after we split the config to multiple configs, there won't be such confusion.
        assert optimizer is not None, "Optimizer is required. Please use QueryProcessorFactory.create_processor() to initialize a QueryProcessor."
        self.optimizer = optimizer

        logger.info(f"Initialized QueryProcessor {self.__class__.__name__}")
        logger.debug(f"QueryProcessor initialized with config: {self.config}")

    def _get_datareader(self, dataset: Set | DataReader) -> DataReader:
        """
        Gets the DataReader for the given dataset.
        """
        # iterate until we reach DataReader
        while isinstance(dataset, Set):
            dataset = dataset._source

        return dataset

    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"

        return hash_for_id(id_str)

    def get_max_quality_plan_id(self, plans: list[PhysicalPlan]) -> str:
        """
        Return the plan_id for the plan with the highest quality in the list of plans.
        """
        max_quality_plan_id, max_quality = None, -1
        for plan in plans:
            if plan.quality > max_quality or max_quality_plan_id is None:
                max_quality_plan_id = plan.plan_id
                max_quality = plan.quality

        return max_quality_plan_id

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

    def execute_plans(
        self, plans: list[PhysicalPlan], max_quality_plan_id: str, num_samples: int | float = float("inf")
    ):
        """
        Execute a given list of plans for num_samples records each. Plans are executed in parallel.
        If any workers are unused, then additional workers are distributed evenly among plans.
        """
        logger.info(f"Executing plans: {plans}")
        # compute number of plans
        num_plans = len(plans)

        # set plan_parallel_workers and workers_per_plan;
        # - plan_parallel_workers controls how many plans are executed in parallel
        # - workers_per_plan controls how many threads are assigned to executing each plan
        plan_parallel_workers, workers_per_plan = None, None
        if self.max_workers <= num_plans:
            plan_parallel_workers = self.max_workers
            workers_per_plan = [1 for _ in range(num_plans)]
        else:
            plan_parallel_workers = num_plans
            workers_per_plan = [(self.max_workers // num_plans) for _ in range(num_plans)]
            idx = 0
            while sum(workers_per_plan) < self.max_workers:
                workers_per_plan[idx] += 1
                idx += 1

        with ThreadPoolExecutor(max_workers=plan_parallel_workers) as executor:
            results = list(executor.map(lambda x: self.execute_plan(**x),
                    [{"plan": plan,
                      "num_samples": num_samples,
                      "plan_workers": plan_workers}
                      for plan, plan_workers in zip(plans, workers_per_plan)],
                )
            )
        # results = list(map(lambda x: self.execute_plan(**x),
        #         [{"plan": plan,
        #             "num_samples": num_samples,
        #             "plan_workers": 1}
        #             for plan in plans],
        #     )
        # )
        # split results into per-plan records and plan stats
        all_records, all_plan_stats = zip(*results)

        # process results to get sample execution data and sentinel plan stats
        all_sample_execution_data, return_records = [], []
        for records, plan_stats, plan in zip(all_records, all_plan_stats, plans):
            # aggregate sentinel est. data
            for operator_stats in plan_stats.operator_stats.values():
                all_sample_execution_data.extend(operator_stats.record_op_stats_lst)

            # if this is the max quality plan for this set of plans, return its results for these records
            if plan.plan_id == max_quality_plan_id:
                return_records = records

        logger.info(f"Done executing plans number: {len(plans)}")
        logger.debug(f"All sample execution data number: {len(all_sample_execution_data)}")
        logger.debug(f"Return records number: {len(return_records)}")
        logger.debug(f"All plan stats number: {len(all_plan_stats)}")
        return all_sample_execution_data, return_records, all_plan_stats
    
    def _execute_best_plan(
        self,
        dataset: Dataset,
        policy: Policy,
        optimizer: Optimizer,
        execution_data: list[RecordOpStats] | None = None,
    ) -> tuple[list[DataRecord], list[PlanStats]]:
        # get the optimal plan according to the optimizer
        plans = optimizer.optimize(dataset, policy)
        final_plan = plans[0]
        # execute the plan
        # TODO: for some reason this is not picking up change to self.max_workers from PipelinedParallelPlanExecutor.__init__()
        records, plan_stats = self.execute_plan(
            plan=final_plan,
            plan_workers=self.max_workers,
        )

        # return the output records and plan stats
        return records, [plan_stats]
    
    def _execute_with_strategy(
        self,
        dataset: Dataset,
        policy: Policy,
        optimizer: Optimizer,
        execution_data: list[RecordOpStats] | None = None,
    ) -> tuple[list[DataRecord], list[PlanStats]]:
        records, plan_stats = [], []
        if optimizer.optimization_strategy_type == OptimizationStrategyType.CONFIDENCE_INTERVAL:
            records, plan_stats = self._execute_confidence_interval_strategy(dataset, policy, optimizer, execution_data)
        else:
            records, plan_stats = self._execute_best_plan(dataset, policy, optimizer, execution_data)
        return records, plan_stats


    def _execute_confidence_interval_strategy(
        self,
        dataset: Dataset,
        policy: Policy,
        optimizer: Optimizer,
        execution_data: list[RecordOpStats] | None = None,
    ) -> tuple[list[DataRecord], list[PlanStats]]:
        # initialize output records and plan stats
        if execution_data is None:
            execution_data = []
        records, plan_stats = [], []

        # get the initial set of optimal plans according to the optimizer
        plans = optimizer.optimize(dataset, policy)
        while len(plans) > 1 and self.scan_start_idx < len(self.datareader):
            # identify the plan with the highest quality in the set
            max_quality_plan_id = self.get_max_quality_plan_id(plans)

            # execute the set of plans for a fixed number of samples
            new_execution_data, new_records, new_plan_stats = self.execute_plans(
                list(plans), max_quality_plan_id, self.num_samples
            )
            records.extend(new_records)
            plan_stats.extend(new_plan_stats)

            if self.scan_start_idx + self.num_samples < len(self.datareader):
                # update cost model and optimizer
                execution_data.extend(new_execution_data)
                cost_model = CostModel(sample_execution_data=execution_data)
                optimizer.update_cost_model(cost_model)

                # get new set of plans
                plans = optimizer.optimize(dataset, policy)

                # update scan start idx
                self.scan_start_idx += self.num_samples

        if self.scan_start_idx < len(self.datareader):
            # execute final plan until end
            final_plan = plans[0]
            new_records, new_plan_stats = self.execute_plan(
                plan=final_plan,
                plan_workers=self.max_workers,
            )
            records.extend(new_records)
            plan_stats.append(new_plan_stats)

        # return the final set of records and plan stats
        return records, plan_stats

    # TODO: consider to support dry_run.
    @abstractmethod
    def execute(self) -> DataRecordCollection:
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")