import hashlib
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

from palimpzest.constants import MAX_ID_CHARS, Model, OptimizationStrategy
from palimpzest.cost_model import CostModel
from palimpzest.dataclasses import PlanStats, RecordOpStats
from palimpzest.datamanager import DataDirectory
from palimpzest.elements.records import DataRecord
from palimpzest.optimizer.optimizer import Optimizer
from palimpzest.optimizer.plan import PhysicalPlan
from palimpzest.policy import Policy
from palimpzest.sets import Set
from palimpzest.utils.model_helpers import getModels


class ExecutionEngine:
    def __init__(
        self,
        num_samples: int = float("inf"),
        scan_start_idx: int = 0,
        nocache: bool = True,  # NOTE: until we properly implement caching, let's set the default to True
        include_baselines: bool = False,
        min_plans: Optional[int] = None,
        verbose: bool = False,
        available_models: List[Model] | None = None,
        allow_bonded_query: bool = True,
        allow_conventional_query: bool = False,
        allow_model_selection: bool = True,
        allow_code_synth: bool = True,
        allow_token_reduction: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.OPTIMAL,
        max_workers: int = 1,
        num_workers_per_thread: int = 1,
        inter_plan_parallelism: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        if self.available_models is None or len(self.available_models) == 0:
            self.available_models = getModels(include_vision=True)
        if self.verbose:
            print("Available models: ", self.available_models)
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.optimization_strategy = optimization_strategy
        self.max_workers = max_workers
        self.num_workers_per_thread = num_workers_per_thread
        self.inter_plan_parallelism = inter_plan_parallelism

        self.datadir = DataDirectory()

    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"

        return hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:MAX_ID_CHARS]

    def clear_cached_responses_and_examples(self):
        """
        Clear cached LLM responses and codegen samples.
        """
        dspy_cache_dir = os.path.join(os.path.expanduser("~"), "cachedir_joblib/joblib/dsp/")
        if os.path.exists(dspy_cache_dir):
            shutil.rmtree(dspy_cache_dir)
        cache = self.datadir.getCacheService()
        cache.rmCache()

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

    def get_parallel_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return max(int(0.8 * multiprocessing.cpu_count()), 1)

    def get_max_quality_plan_id(self, plans: List[PhysicalPlan]) -> str:
        """
        Return the plan_id for the plan with the highest quality in the list of plans.
        """
        max_quality_plan_id, max_quality = None, -1
        for plan in plans:
            if plan.quality > max_quality or max_quality_plan_id is None:
                max_quality_plan_id = plan.plan_id
                max_quality = plan.quality

        return max_quality_plan_id

    def aggregate_plan_stats(self, plan_stats: List[PlanStats]) -> Dict[str, PlanStats]:
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
        self, plans: List[PhysicalPlan], max_quality_plan_id: str, num_samples: Union[int, float] = float("inf")
    ):
        """
        Execute a given list of plans for num_samples records each, using whatever parallelism is available.
        """
        # compute number of plans
        num_plans = len(plans)

        # execute plans using any parallelism provided by the user or system
        max_workers = (
            self.max_workers
            if not self.inter_plan_parallelism
            else max(self.max_workers, self.get_parallel_max_workers())
        )
        max_workers = min(max_workers, num_plans)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda x: self.execute_plan(**x),
                    [
                        {"plan": plan, "num_samples": num_samples, "max_workers": self.num_workers_per_thread}
                        for plan in plans
                    ],
                )
            )
        # results = list(map(lambda x: self.execute_plan(**x),
        #         [{"plan": plan,
        #             "num_samples": num_samples,
        #             "max_workers": 1}
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

        return all_sample_execution_data, return_records, plan_stats

    def execute_optimal_strategy(
        self,
        dataset: Set,
        optimizer: Optimizer,
        execution_data: List[RecordOpStats] | None = None,
    ) -> Tuple[List[DataRecord], List[PlanStats]]:
        # get the optimal plan according to the optimizer
        if execution_data is None:
            execution_data = []
        plans = optimizer.optimize(dataset)
        final_plan = plans[0]

        # execute the plan
        records, plan_stats = self.execute_plan(
            plan=final_plan,
            max_workers=self.max_workers,
        )

        # return the output records and plan stats
        return records, [plan_stats]

    def execute_confidence_interval_strategy(
        self,
        dataset: Set,
        optimizer: Optimizer,
        execution_data: List[RecordOpStats] | None = None,
    ) -> Tuple[List[DataRecord], List[PlanStats]]:
        # initialize output records and plan stats
        if execution_data is None:
            execution_data = []
        records, plan_stats = [], []

        # get total number of input records in the datasource
        datasource = self.datadir.getRegisteredDataset(self.source_dataset_id)
        datasource_len = len(datasource)

        # get the initial set of optimal plans according to the optimizer
        plans = optimizer.optimize(dataset)
        while len(plans) > 1 and self.scan_start_idx < datasource_len:
            # identify the plan with the highest quality in the set
            max_quality_plan_id = self.get_max_quality_plan_id(plans)

            # execute the set of plans for a fixed number of samples
            new_execution_data, new_records, new_plan_stats = self.execute_plans(
                list(plans), max_quality_plan_id, self.num_samples
            )
            records.extend(new_records)
            plan_stats.extend(new_plan_stats)

            if self.scan_start_idx + self.num_samples < datasource_len:
                # update cost model and optimizer
                execution_data.extend(new_execution_data)
                cost_model = CostModel(
                    source_dataset_id=self.source_dataset_id,
                    sample_execution_data=execution_data,
                )
                optimizer.update_cost_model(cost_model)

                # get new set of plans
                plans = optimizer.optimize(dataset)

                # update scan start idx
                self.scan_start_idx += self.num_samples

        if self.scan_start_idx < datasource_len:
            # execute final plan until end
            final_plan = plans[0]
            new_records, new_plan_stats = self.execute_plan(
                plan=final_plan,
                max_workers=self.max_workers,
            )
            records.extend(new_records)
            plan_stats.append(new_plan_stats)

        # return the final set of records and plan stats
        return records, plan_stats

    def execute_plan(
        self, plan: PhysicalPlan, num_samples: Union[int, float] = float("inf"), max_workers: Optional[int] = None
    ):
        """Execute the given plan and return the output records and plan stats."""
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")

    def execute(self, dataset: Set, policy: Policy):
        """
        Execute the workload specified by the given dataset according to the policy provided by the user.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
