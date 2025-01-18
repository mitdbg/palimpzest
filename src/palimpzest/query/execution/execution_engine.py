import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from palimpzest.constants import Model, OptimizationStrategy
from palimpzest.core.data.dataclasses import PlanStats, RecordOpStats
from palimpzest.core.data.datasources import DataSource, ValidationDataSource
from palimpzest.core.elements.records import DataRecord
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import Policy
from palimpzest.query.optimizer.cost_model import CostModel
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.plan import PhysicalPlan
from palimpzest.sets import Dataset, Set
from palimpzest.utils.hash_helpers import hash_for_id
from palimpzest.utils.model_helpers import get_models


class ExecutionEngine:
    def __init__(
        self,
        datasource: DataSource,
        num_samples: int = float("inf"),
        scan_start_idx: int = 0,
        nocache: bool = True,  # NOTE: until we properly implement caching, let's set the default to True
        include_baselines: bool = False,
        min_plans: int | None = None,
        verbose: bool = False,
        available_models: list[Model] | None = None,
        allow_bonded_query: bool = True,
        allow_conventional_query: bool = False,
        allow_model_selection: bool = True,
        allow_code_synth: bool = True,
        allow_token_reduction: bool = False,
        allow_rag_reduction: bool = True,
        allow_mixtures: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.PARETO,
        max_workers: int | None = None,
        num_workers_per_plan: int = 1,
        *args,
        **kwargs,
    ) -> None:
        self.num_samples = num_samples
        self.scan_start_idx = scan_start_idx
        self.nocache = nocache
        if not self.nocache:
            raise NotImplementedError("Caching is not yet implemented! Please set nocache=True.")
        self.include_baselines = include_baselines
        self.min_plans = min_plans
        self.verbose = verbose
        self.available_models = available_models
        if self.available_models is None or len(self.available_models) == 0:
            self.available_models = get_models(include_vision=True)
        if self.verbose:
            print("Available models: ", self.available_models)
        self.allow_bonded_query = allow_bonded_query
        self.allow_conventional_query = allow_conventional_query
        self.allow_model_selection = allow_model_selection
        self.allow_code_synth = allow_code_synth
        self.allow_token_reduction = allow_token_reduction
        self.allow_rag_reduction = allow_rag_reduction
        self.allow_mixtures = allow_mixtures
        self.optimization_strategy = optimization_strategy
        self.max_workers = max_workers
        self.num_workers_per_plan = num_workers_per_plan

        self.datadir = DataDirectory()

        # datasource; should be set by execute() with call to get_datasource()
        self.datasource = datasource
        self.using_validation_data = isinstance(self.datasource, ValidationDataSource)


    def execution_id(self) -> str:
        """
        Hash of the class parameters.
        """
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"

        return hash_for_id(id_str)

    def clear_cached_examples(self):
        """
        Clear cached codegen samples.
        """
        cache = self.datadir.get_cache_service()
        cache.rm_cache()

    def get_parallel_max_workers(self):
        # for now, return the number of system CPUs;
        # in the future, we may want to consider the models the user has access to
        # and whether or not they will encounter rate-limits. If they will, we should
        # set the max workers in a manner that is designed to avoid hitting them.
        # Doing this "right" may require considering their logical, physical plan,
        # and tier status with LLM providers. It may also be worth dynamically
        # changing the max_workers in response to 429 errors.
        return max(int(0.8 * multiprocessing.cpu_count()), 1)

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

        return all_sample_execution_data, return_records, all_plan_stats

    def execute_strategy(
        self,
        dataset: Set,
        policy: Policy,
        optimizer: Optimizer,
        execution_data: list[RecordOpStats] | None = None,
    ) -> tuple[list[DataRecord], list[PlanStats]]:
        if execution_data is None:
            execution_data = []

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

    def execute_confidence_interval_strategy(
        self,
        dataset: Set,
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
        while len(plans) > 1 and self.scan_start_idx < len(self.datasource):
            # identify the plan with the highest quality in the set
            max_quality_plan_id = self.get_max_quality_plan_id(plans)

            # execute the set of plans for a fixed number of samples
            new_execution_data, new_records, new_plan_stats = self.execute_plans(
                list(plans), max_quality_plan_id, self.num_samples
            )
            records.extend(new_records)
            plan_stats.extend(new_plan_stats)

            if self.scan_start_idx + self.num_samples < len(self.datasource):
                # update cost model and optimizer
                execution_data.extend(new_execution_data)
                cost_model = CostModel(sample_execution_data=execution_data)
                optimizer.update_cost_model(cost_model)

                # get new set of plans
                plans = optimizer.optimize(dataset, policy)

                # update scan start idx
                self.scan_start_idx += self.num_samples

        if self.scan_start_idx < len(self.datasource):
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


    def execute_plan(self, plan: PhysicalPlan, num_samples: int | float = float("inf"), plan_workers: int = 1):
        """Execute the given plan and return the output records and plan stats."""
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")


    def execute(self, dataset: Dataset, policy: Policy):
        """
        Execute the workload specified by the given dataset according to the policy provided by the user.
        """
        raise NotImplementedError("Abstract method to be overwritten by sub-classes")
