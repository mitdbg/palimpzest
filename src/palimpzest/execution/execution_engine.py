from palimpzest.constants import ExecutionStrategy, Model, PlanType, PlanPruningStrategy, MAX_UUID_CHARS
from palimpzest.cost_estimator import CostEstimator
from palimpzest.datamanager import DataDirectory
from palimpzest.planner import PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.pruning import ParetoPruningStrategy
from palimpzest.sets import Set
from palimpzest.utils import getModels

from typing import List, Optional

import hashlib
import multiprocessing


class ExecutionEngine:
    def __init__(self,
            num_samples: int=20,
            scan_start_idx: int=0,
            nocache: bool=False,
            include_baselines: bool=False,
            min_plans: Optional[int] = None,
            verbose: bool = False,
            available_models: List[Model] = [],
            allow_bonded_query: bool=True,
            allow_conventional_query: bool=False,
            allow_model_selection: bool=True,
            allow_code_synth: bool=True,
            allow_token_reduction: bool=True,
            plan_pruning_strategy: Optional[PlanPruningStrategy]=PlanPruningStrategy.PARETO,
            confidence_interval_pruning: Optional[bool]=True,
            execution_strategy: bool=ExecutionStrategy.SINGLE_THREADED,
            max_workers: int=1,
            *args, **kwargs
        ) -> None:
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        if not available_models:
            self.available_models = getModels()
        print("Available models: ", self.available_models)
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.plan_pruning_strategy = plan_pruning_strategy
        self.confidence_interval_pruning = confidence_interval_pruning
        self.execution_strategy = execution_strategy
        self.max_workers = max_workers
        if self.max_workers is None and self.execution_strategy == ExecutionStrategy.PARALLEL:
            self.max_workers = self.set_max_workers()
        self.datadir = DataDirectory()

    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        uuid_str = ""
        for attr, value in self.__dict__.keys():
            if not attr.startswith("_"):
                uuid_str += f"{attr}={value},"

        return hashlib.sha256(uuid_str.encode("utf-8")).hexdigest()[:MAX_UUID_CHARS]

    def set_source_dataset_id(self, dataset: Set) -> str:
        """
        Sets the dataset_id of the DataSource for the given dataset.
        """
        # iterate until we reach DataSource
        while isinstance(dataset, Set):
            dataset = dataset._source

        # throw an exception if datasource is not registered with PZ
        _ = self.datadir.getRegisteredDataset(dataset.dataset_id)

        # set the source dataset id
        self.source_dataset_id = dataset.dataset_id

    def set_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return multiprocessing.cpu_count()

    def get_pruning_strategy(self, cost_estimator: CostEstimator):
        if self.plan_pruning_strategy == PlanPruningStrategy.PARETO:
            return ParetoPruningStrategy(cost_estimator)

        return None

    def execute_plan(self, plan: PhysicalPlan,
                     plan_type: PlanType = PlanType.FINAL,
                     plan_idx: Optional[int] = None,
                     max_workers: Optional[int] = None):
        """Execute the given plan and return the output records and plan stats."""
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")

    def execute(self, dataset: Set, policy: Policy):
        """
        Execute the workload specified by the given dataset according to the policy provided by the user.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
