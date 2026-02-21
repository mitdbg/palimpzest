"""
Modified QueryProcessor with COMPLETE Filter Selectivity Optimization

This shows how to integrate filter_selectivity.py into query_processor.py.

Changes from original:
1. Import filter_selectivity module
2. Add optional parameters for filter optimization
3. Insert filter selectivity estimation BETWEEN lines 120-122
4. Option to generate/prune filter orderings or just reorder

Place filter_selectivity.py at: src/palimpzest/query/optimizer/filter_selectivity.py
"""

import logging

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.models import ExecutionStats, PlanStats
from palimpzest.policy import Policy
from palimpzest.query.execution.execution_strategy import ExecutionStrategy, SentinelExecutionStrategy
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel

# ============================================================================
# NEW IMPORTS - Filter Selectivity Module
# ============================================================================
from palimpzest.query.optimizer.filter_selectivity import (
    SelectivityEstimationResult,
    estimate_filter_selectivity_for_sentinel_plan,
    generate_and_prune_filter_orderings,
    optimize_filter_ordering,
)
from palimpzest.query.optimizer.optimizer import Optimizer
from palimpzest.query.optimizer.optimizer_strategy_type import OptimizationStrategyType
from palimpzest.query.optimizer.plan import SentinelPlan
from palimpzest.utils.hash_helpers import hash_for_id
from palimpzest.validator.validator import Validator

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes queries through the complete pipeline with filter optimization.
    
    New features:
    - Estimates filter selectivity on sample data
    - Reorders filters to place most selective first
    - Optionally generates and prunes filter ordering variants
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
        # ====================================================================
        # NEW PARAMETERS for filter selectivity optimization
        # ====================================================================
        estimate_filter_selectivity: bool = True,   # Enable selectivity estimation
        filter_selectivity_samples: int = 10,        # Samples for estimation
        reorder_filters: bool = True,                # Reorder filters by selectivity
        prune_filter_orderings: bool = False,        # Generate & prune all orderings
        max_filter_permutations: int = 24,           # Max permutations to generate
        # ====================================================================
        **kwargs,
    ):
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
        
        # NEW: Filter selectivity settings
        self.estimate_filter_selectivity = estimate_filter_selectivity
        self.filter_selectivity_samples = filter_selectivity_samples
        self.reorder_filters = reorder_filters
        self.prune_filter_orderings = prune_filter_orderings
        self.max_filter_permutations = max_filter_permutations
        
        # Store results for external access
        self._filter_selectivity_result: SelectivityEstimationResult | None = None

        if self.verbose:
            print("Available models:", self.available_models)
            if self.estimate_filter_selectivity:
                print(f"Filter selectivity enabled ({self.filter_selectivity_samples} samples)")

        logger.info(f"Initialized QueryProcessor {self.__class__.__name__}")

    def execution_id(self) -> str:
        id_str = ""
        for attr, value in self.__dict__.items():
            if not attr.startswith("_"):
                id_str += f"{attr}={value},"
        return hash_for_id(id_str)

    def _create_sentinel_plan(self, train_dataset: dict[str, Dataset] | None) -> SentinelPlan:
        """Generates and returns a SentinelPlan for the given dataset."""
        optimizer = self.optimizer.deepcopy_clean()
        optimizer.update_strategy(OptimizationStrategyType.SENTINEL)

        dataset = self.dataset.copy()
        if train_dataset is not None:
            dataset._set_root_datasets(train_dataset)
            dataset._generate_unique_logical_op_ids()

        sentinel_plans = optimizer.optimize(dataset)
        sentinel_plan = sentinel_plans[0]
        return sentinel_plan

    def _execute_best_plan(self, dataset: Dataset, optimizer: Optimizer) -> tuple[list[DataRecord], list[PlanStats]]:
        plans = optimizer.optimize(dataset)
        final_plan = plans[0]
        records, plan_stats = self.execution_strategy.execute_plan(plan=final_plan)
        return records, [plan_stats]
    
    # ========================================================================
    # NEW METHOD: Filter selectivity estimation and optimization
    # ========================================================================
    def _optimize_filter_ordering(
        self,
        sentinel_plan: SentinelPlan,
        train_dataset: dict[str, Dataset],
    ) -> SentinelPlan:
        """
        Estimate filter selectivity and optimize filter ordering.
        
        This implements BOTH steps from the task:
        
        Step 1: Use sentinel_plan to measure filter selectivity
                - Run each filter independently on sample records
                - Compute selectivity (fraction that pass)
                
        Step 2: Optimize filter ordering
                - Option A: Create single optimally-ordered plan (reorder_filters=True)
                - Option B: Generate all orderings and prune non-optimal (prune_filter_orderings=True)
        
        Returns:
            Optimized SentinelPlan with filters reordered by selectivity
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("FILTER SELECTIVITY OPTIMIZATION")
            print("=" * 70)
        
        # ================================================================
        # STEP 1: Estimate filter selectivity
        # ================================================================
        self._filter_selectivity_result = estimate_filter_selectivity_for_sentinel_plan(
            sentinel_plan=sentinel_plan,
            train_dataset=train_dataset,
            num_samples=self.filter_selectivity_samples,
            verbose=self.verbose,
        )
        
        # If no filters found, return original plan
        if len(self._filter_selectivity_result.filter_stats) == 0:
            if self.verbose:
                print("No filters found in plan - skipping optimization")
            return sentinel_plan
        
        # ================================================================
        # STEP 2: Optimize filter ordering
        # ================================================================
        
        if self.prune_filter_orderings:
            # Option B: Generate all orderings and prune non-optimal
            if self.verbose:
                print("\nGenerating and pruning filter orderings...")
            
            optimal_plans = generate_and_prune_filter_orderings(
                sentinel_plan=sentinel_plan,
                selectivity_result=self._filter_selectivity_result,
                max_permutations=self.max_filter_permutations,
                verbose=self.verbose,
            )
            
            # Return the first optimal plan
            optimized_plan = optimal_plans[0] if optimal_plans else sentinel_plan
            
        elif self.reorder_filters:
            # Option A: Create single optimally-ordered plan
            optimized_plan = optimize_filter_ordering(
                sentinel_plan=sentinel_plan,
                selectivity_result=self._filter_selectivity_result,
                verbose=self.verbose,
            )
        else:
            # Just estimate selectivity, don't reorder
            self._filter_selectivity_result.print_summary(verbose=self.verbose)
            optimized_plan = sentinel_plan
        
        return optimized_plan
    
    def get_filter_selectivity_result(self) -> SelectivityEstimationResult | None:
        """Get the filter selectivity results from the last execution."""
        return self._filter_selectivity_result

    def execute(self) -> DataRecordCollection:
        logger.info(f"Executing {self.__class__.__name__}")

        execution_stats = ExecutionStats(execution_id=self.execution_id())
        execution_stats.start()

        if self.validator is not None:
            # ==============================================================
            # LINE 120: Create sentinel plan (ORIGINAL)
            # ==============================================================
            sentinel_plan = self._create_sentinel_plan(self.train_dataset)
        if self.estimate_filter_selectivity:
            # Get training dataset for selectivity estimation
            if self.train_dataset is not None:
                selectivity_train_dataset = self.train_dataset
            else:
                selectivity_train_dataset = self.dataset._get_root_datasets()

            # ==============================================================
            # NEW CODE - INSERT BETWEEN LINES 120-122
            # Filter selectivity estimation and optimization
            # ==============================================================
            if self.estimate_filter_selectivity:
                # Get training dataset
                if self.train_dataset is not None:
                    selectivity_train_dataset = self.train_dataset
                else:
                    selectivity_train_dataset = self.dataset._get_root_datasets()
                
                # Optimize filter ordering based on selectivity
                sentinel_plan = self._optimize_filter_ordering(
                    sentinel_plan=sentinel_plan,
                    train_dataset=selectivity_train_dataset,
                )
            # ==============================================================
            # END NEW CODE
            # ==============================================================

            # ==============================================================
            # LINE 122+: Execute sentinel plan (ORIGINAL)
            # ==============================================================
            if self.train_dataset is not None:
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(
                    sentinel_plan, self.train_dataset, self.validator
                )
            else:
                train_dataset = self.dataset._get_root_datasets()
                sentinel_plan_stats = self.sentinel_execution_strategy.execute_sentinel_plan(
                    sentinel_plan, train_dataset, self.validator
                )

            execution_stats.add_plan_stats(sentinel_plan_stats)
            execution_stats.finish_optimization()

            self.optimizer = self.optimizer.deepcopy_clean()
            cost_model = SampleBasedCostModel(sentinel_plan_stats, self.verbose)
            self.optimizer.update_cost_model(cost_model)

        records, plan_stats = self._execute_best_plan(self.dataset, self.optimizer)

        execution_stats.add_plan_stats(plan_stats)
        execution_stats.finish()

        result = DataRecordCollection(records, execution_stats=execution_stats)
        logger.info(f"Done executing {self.__class__.__name__}")

        return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
Example usage with Palimpzest:

    import palimpzest as pz
    
    # Define pipeline with multiple filters
    dataset = pz.Dataset("testdata/enron-tiny/")
    dataset = dataset.sem_add_columns(email_cols)
    dataset = dataset.sem_filter("The email was sent in July")      # Filter A
    dataset = dataset.sem_filter("The email is about holidays")    # Filter B
    
    # Run with filter optimization enabled (default)
    config = pz.QueryProcessorConfig(
        policy=pz.MinCost(),
        verbose=True,
        
        # Filter selectivity options:
        estimate_filter_selectivity=True,   # Enable selectivity estimation
        filter_selectivity_samples=10,       # Number of samples
        reorder_filters=True,                # Reorder filters optimally
        prune_filter_orderings=False,        # Set True to generate all permutations
    )
    
    output = dataset.run(config)
    
    # The system will:
    # 1. Estimate selectivity: A=80% pass, B=20% pass
    # 2. Reorder: B (20%) → A (80%) for optimal execution
    # 3. Result: Fewer records processed by downstream operators
"""
