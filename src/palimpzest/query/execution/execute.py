from palimpzest.constants import Model, OptimizationStrategy
from palimpzest.core.data.datasources import DataSource
from palimpzest.datamanager.datamanager import DataDirectory
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_engine import ExecutionEngine
from palimpzest.query.execution.nosentinel_execution import NoSentinelSequentialSingleThreadExecution
from palimpzest.sets import Set


class Execute:
    @classmethod
    def get_datasource(cls, dataset: Set | DataSource) -> str:
        """
        Gets the DataSource for the given dataset.
        """
        # iterate until we reach DataSource
        while isinstance(dataset, Set):
            dataset = dataset._source

        # this will throw an exception if datasource is not registered with PZ
        return DataDirectory().get_registered_dataset(dataset.dataset_id)

    def __new__(
        cls,
        dataset: Set,
        policy: Policy,
        num_samples: int = 20,
        nocache: bool = False,
        include_baselines: bool = False,
        min_plans: int | None = None,
        max_workers: int = 1,
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
        execution_engine: ExecutionEngine = NoSentinelSequentialSingleThreadExecution,
        *args,
        **kwargs,
    ):
        if available_models is None:
            available_models = []
        return execution_engine(
            *args,
            **kwargs,
            datasource=cls.get_datasource(dataset),
            num_samples=num_samples,
            nocache=nocache,
            include_baselines=include_baselines,
            min_plans=min_plans,
            max_workers=max_workers,
            verbose=verbose,
            available_models=available_models,
            allow_bonded_query=allow_bonded_query,
            allow_conventional_query=allow_conventional_query,
            allow_code_synth=allow_code_synth,
            allow_model_selection=allow_model_selection,
            allow_token_reduction=allow_token_reduction,
            allow_rag_reduction=allow_rag_reduction,
            allow_mixtures=allow_mixtures,
            optimization_strategy=optimization_strategy,
        ).execute(dataset=dataset, policy=policy)
