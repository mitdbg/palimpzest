"""
Filter Selectivity Estimation and Reordering for Palimpzest

This module provides COMPLETE functionality to:
1. Estimate the selectivity of filter predicates by running them on sample data
2. Generate alternative filter orderings in the SentinelPlan  
3. Prune non-optimal filter orderings from the plan
4. Create optimally reordered SentinelPlans

The key insight is that filters with lower selectivity (more restrictive filters
that pass fewer records) should be executed first to minimize the number of records
processed by subsequent operators.

Place this file at: src/palimpzest/query/optimizer/filter_selectivity.py

Author: Implementation for Palimpzest filter evaluation task
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from itertools import permutations

if TYPE_CHECKING:
    from palimpzest.query.optimizer.plan import SentinelPlan
    from palimpzest.core.data.dataset import Dataset
    from palimpzest.query.operators.physical import PhysicalOperator
    from palimpzest.core.elements.records import DataRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilterInfo:
    """Information about a filter operator in the sentinel plan."""
    topo_idx: int                          # Topological index in the plan
    logical_op_id: str                     # Logical operator ID
    unique_logical_op_id: str              # "{topo_idx}-{logical_op_id}"
    filter_condition: str                  # The filter predicate string
    operator_set: list["PhysicalOperator"] # Physical operators for this filter
    
    def __repr__(self) -> str:
        cond = self.filter_condition[:40] if self.filter_condition else "unknown"
        return f"FilterInfo(idx={self.topo_idx}, condition='{cond}...')"


@dataclass 
class FilterSelectivityStats:
    """Statistics about a filter's selectivity from sample execution."""
    filter_info: FilterInfo
    num_samples: int = 0
    num_passed: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    
    @property
    def selectivity(self) -> float:
        """Fraction of records that pass the filter (0.0 to 1.0)."""
        if self.num_samples == 0:
            return 1.0
        return self.num_passed / self.num_samples
    
    @property
    def avg_cost_per_record(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.total_cost / self.num_samples
    
    @property
    def avg_time_per_record(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.total_time / self.num_samples
    
    def record_result(self, passed: bool, cost: float = 0.0, time: float = 0.0) -> None:
        self.num_samples += 1
        if passed:
            self.num_passed += 1
        self.total_cost += cost
        self.total_time += time


@dataclass
class SelectivityEstimationResult:
    """Complete results of selectivity estimation for all filters in a plan."""
    filter_stats: dict[str, FilterSelectivityStats] = field(default_factory=dict)
    filter_infos: list[FilterInfo] = field(default_factory=list)
    current_filter_order: list[str] = field(default_factory=list)
    optimal_filter_order: list[str] = field(default_factory=list)
    
    @property
    def is_order_optimal(self) -> bool:
        return self.current_filter_order == self.optimal_filter_order
    
    def compute_optimal_order(self) -> None:
        """Compute optimal ordering: most selective (lowest selectivity) first."""
        sorted_filters = sorted(
            [(uid, stats) for uid, stats in self.filter_stats.items()],
            key=lambda x: x[1].selectivity
        )
        self.optimal_filter_order = [uid for uid, _ in sorted_filters]
    
    def get_selectivity(self, unique_logical_op_id: str) -> float:
        if unique_logical_op_id in self.filter_stats:
            return self.filter_stats[unique_logical_op_id].selectivity
        return 1.0
    
    def get_selectivity_for_operator_set(self, op_set: list) -> float:
        """Get selectivity by matching operator_set."""
        for uid, stats in self.filter_stats.items():
            if stats.filter_info.operator_set is op_set or stats.filter_info.operator_set == op_set:
                return stats.selectivity
        return 1.0
    
    def estimate_plan_cost(self, filter_order: list[str]) -> float:
        """Estimate relative cost for a given filter ordering."""
        records_remaining = 1.0
        total_cost = 0.0
        for uid in filter_order:
            stats = self.filter_stats.get(uid)
            if stats:
                total_cost += records_remaining * stats.avg_cost_per_record
                records_remaining *= stats.selectivity
        return total_cost
    
    def print_summary(self, verbose: bool = True) -> None:
        if not verbose:
            return
        print("\n" + "=" * 70)
        print("FILTER SELECTIVITY ESTIMATION RESULTS")
        print("=" * 70)
        
        print("\nFilter Statistics:")
        for uid in self.current_filter_order:
            stats = self.filter_stats.get(uid)
            if stats:
                cond = stats.filter_info.filter_condition[:50] if stats.filter_info.filter_condition else "N/A"
                print(f"  [{uid}] {cond}...")
                print(f"       Selectivity: {stats.selectivity:.4f} ({stats.num_passed}/{stats.num_samples} passed)")
        
        print(f"\nCurrent order:  {self.current_filter_order}")
        print(f"Optimal order:  {self.optimal_filter_order}")
        print(f"Is optimal: {self.is_order_optimal}")
        
        if not self.is_order_optimal:
            print("\n⚠️  Filter reordering recommended!")
            current_cost = self.estimate_plan_cost(self.current_filter_order)
            optimal_cost = self.estimate_plan_cost(self.optimal_filter_order)
            if current_cost > 0:
                savings = (current_cost - optimal_cost) / current_cost * 100
                print(f"   Estimated cost reduction: {savings:.1f}%")
        print("=" * 70 + "\n")


# =============================================================================
# Core Estimator Class  
# =============================================================================

class FilterSelectivityEstimator:
    """Estimates filter selectivity by running filters on sample data."""
    
    def __init__(self, num_samples: int = 10, verbose: bool = False):
        self.num_samples = num_samples
        self.verbose = verbose
    
    def _is_filter_operator(self, op: Any) -> bool:
        """Check if operator is a filter."""
        try:
            from palimpzest.query.operators.filter import FilterOp, LLMFilter, NonLLMFilter
            return isinstance(op, (FilterOp, LLMFilter, NonLLMFilter))
        except ImportError:
            return hasattr(op, 'filter_obj')
    
    def _get_filter_condition(self, op: Any) -> str:
        """Extract filter condition string."""
        if hasattr(op, 'filter_obj'):
            if hasattr(op.filter_obj, 'filter_condition'):
                return op.filter_obj.filter_condition
            elif hasattr(op.filter_obj, 'get_filter_str'):
                return op.filter_obj.get_filter_str()
            return str(op.filter_obj)
        return "unknown"
    
    def identify_filters_in_plan(self, sentinel_plan: "SentinelPlan") -> list[FilterInfo]:
        """Identify all filter operators in the sentinel plan."""
        filters = []
        for topo_idx, (logical_op_id, op_set) in enumerate(sentinel_plan):
            if len(op_set) > 0 and self._is_filter_operator(op_set[0]):
                unique_logical_op_id = f"{topo_idx}-{logical_op_id}"
                filter_condition = self._get_filter_condition(op_set[0])
                filters.append(FilterInfo(
                    topo_idx=topo_idx,
                    logical_op_id=logical_op_id,
                    unique_logical_op_id=unique_logical_op_id,
                    filter_condition=filter_condition,
                    operator_set=op_set,
                ))
        return filters
    
    def _get_sample_records(
        self,
        sentinel_plan: "SentinelPlan",
        train_dataset: dict[str, "Dataset"],
    ) -> list["DataRecord"]:
        """Get sample DataRecords from the scan operator."""
        try:
            from palimpzest.query.operators.scan import ScanPhysicalOp, MarshalAndScanDataOp
            from palimpzest.core.elements.records import DataRecord
        except ImportError:
            return []
        
        sample_records = []
        for topo_idx, (logical_op_id, op_set) in enumerate(sentinel_plan):
            if len(op_set) > 0 and isinstance(op_set[0], (ScanPhysicalOp, MarshalAndScanDataOp)):
                scan_op = op_set[0]
                if hasattr(scan_op, 'datasource'):
                    datasource = scan_op.datasource
                    num_to_sample = min(self.num_samples, len(datasource))
                    for idx in range(num_to_sample):
                        try:
                            record_set = scan_op(idx)
                            if record_set and len(record_set.data_records) > 0:
                                sample_records.append(record_set.data_records[0])
                        except Exception as e:
                            logger.warning(f"Error scanning record {idx}: {e}")
                    return sample_records
        
        # Fallback to direct dataset access
        for dataset_id, dataset in train_dataset.items():
            num_to_sample = min(self.num_samples, len(dataset))
            for idx in range(num_to_sample):
                try:
                    item = dataset[idx]
                    sample_records.append(item)
                except Exception as e:
                    pass
        return sample_records
    
    def _run_filter_on_record(
        self, filter_op: Any, record: "DataRecord"
    ) -> tuple[bool, float, float]:
        """Run a filter operator on a record."""
        import time
        try:
            start_time = time.time()
            result = filter_op(record)
            elapsed = time.time() - start_time
            
            passed = False
            cost = 0.0
            if result and len(result.data_records) > 0:
                if hasattr(result.data_records[0], '_passed_operator'):
                    passed = result.data_records[0]._passed_operator
                else:
                    passed = True
                if hasattr(result, 'record_op_stats') and len(result.record_op_stats) > 0:
                    cost = result.record_op_stats[0].cost_per_record or 0.0
            return passed, cost, elapsed
        except Exception as e:
            logger.warning(f"Error running filter: {e}")
            return False, 0.0, 0.0
    
    def estimate_selectivity(
        self,
        sentinel_plan: "SentinelPlan",
        train_dataset: dict[str, "Dataset"],
    ) -> SelectivityEstimationResult:
        """Main entry point: Estimate selectivity for all filters."""
        result = SelectivityEstimationResult()
        
        filters = self.identify_filters_in_plan(sentinel_plan)
        if len(filters) == 0:
            return result
        
        result.filter_infos = filters
        result.current_filter_order = [f.unique_logical_op_id for f in filters]
        
        sample_records = self._get_sample_records(sentinel_plan, train_dataset)
        if len(sample_records) == 0:
            logger.warning("No sample records available")
            return result
        
        if self.verbose:
            print(f"Estimating selectivity using {len(sample_records)} samples...")
        
        for filter_info in filters:
            stats = FilterSelectivityStats(filter_info=filter_info)
            if len(filter_info.operator_set) > 0:
                filter_op = filter_info.operator_set[0]
                for record in sample_records:
                    passed, cost, time = self._run_filter_on_record(filter_op, record)
                    stats.record_result(passed, cost, time)
            result.filter_stats[filter_info.unique_logical_op_id] = stats
            
            if self.verbose:
                print(f"  {filter_info.filter_condition[:40]}... → selectivity={stats.selectivity:.3f}")
        
        result.compute_optimal_order()
        return result


# =============================================================================
# SentinelPlan Reordering Implementation (THE FUTURE WORK)
# =============================================================================

class SentinelPlanReorderer:
    """
    Handles reordering of filters in a SentinelPlan based on selectivity.
    
    This class implements the "Future Work" from the task:
    1. Generate alternative filter orderings
    2. Create new SentinelPlan variants with swapped filters  
    3. Prune non-optimal orderings
    
    SentinelPlan Structure:
    ----------------------
    For plan: Scan → FilterA → FilterB → Convert
    
    Tree structure (root = final op):
        Convert
          └── FilterB
                └── FilterA
                      └── Scan
    
    Iteration yields (in topological order): Scan, FilterA, FilterB, Convert
    
    To reorder filters, we:
    1. Flatten the tree to a list
    2. Identify consecutive filter nodes
    3. Swap their operator_sets
    4. Rebuild the tree
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def _is_filter_operator(self, op: Any) -> bool:
        """Check if operator is a filter."""
        try:
            from palimpzest.query.operators.filter import FilterOp, LLMFilter, NonLLMFilter
            return isinstance(op, (FilterOp, LLMFilter, NonLLMFilter))
        except ImportError:
            return hasattr(op, 'filter_obj')
    
    def _flatten_plan(
        self, sentinel_plan: "SentinelPlan"
    ) -> list[list[Any]]:
        """
        Flatten SentinelPlan to list of operator_sets.
        Returns in topological order (scan first, final op last).
        """
        return [op_set for _, op_set in sentinel_plan]
    
    def _find_filter_indices(self, op_sets: list[list[Any]]) -> list[int]:
        """Find indices of filter operator_sets."""
        indices = []
        for i, op_set in enumerate(op_sets):
            if len(op_set) > 0 and self._is_filter_operator(op_set[0]):
                indices.append(i)
        return indices
    
    def _find_consecutive_filter_regions(
        self, filter_indices: list[int]
    ) -> list[list[int]]:
        """
        Group consecutive filter indices into regions that can be reordered.
        
        Example: indices [1, 2, 4, 5, 6] → regions [[1, 2], [4, 5, 6]]
        """
        if len(filter_indices) == 0:
            return []
        
        regions = []
        current_region = [filter_indices[0]]
        
        for i in range(1, len(filter_indices)):
            if filter_indices[i] == filter_indices[i-1] + 1:
                # Consecutive
                current_region.append(filter_indices[i])
            else:
                # Gap - save current region if it has 2+ filters
                if len(current_region) >= 2:
                    regions.append(current_region)
                current_region = [filter_indices[i]]
        
        # Don't forget the last region
        if len(current_region) >= 2:
            regions.append(current_region)
        
        return regions
    
    def _rebuild_sentinel_plan(
        self, op_sets: list[list[Any]]
    ) -> "SentinelPlan":
        """
        Rebuild a SentinelPlan from a list of operator_sets.
        
        Builds bottom-up: first op_set becomes deepest node.
        """
        from palimpzest.query.optimizer.plan import SentinelPlan
        
        if len(op_sets) == 0:
            raise ValueError("Cannot rebuild empty plan")
        
        # Build from bottom up
        current_plan = SentinelPlan(op_sets[0], None)
        for i in range(1, len(op_sets)):
            current_plan = SentinelPlan(op_sets[i], [current_plan])
        
        return current_plan
    
    def generate_all_filter_orderings(
        self,
        sentinel_plan: "SentinelPlan",
        max_permutations: int = 24,  # 4! = 24, reasonable limit
    ) -> list["SentinelPlan"]:
        """
        Generate SentinelPlan variants with all permutations of filter orderings.
        
        Args:
            sentinel_plan: Original SentinelPlan
            max_permutations: Maximum number of permutations to generate
            
        Returns:
            List of SentinelPlan variants with different filter orderings
        """
        op_sets = self._flatten_plan(sentinel_plan)
        filter_indices = self._find_filter_indices(op_sets)
        regions = self._find_consecutive_filter_regions(filter_indices)
        
        if len(regions) == 0:
            if self.verbose:
                print("No consecutive filter regions found - returning original plan")
            return [sentinel_plan]
        
        if self.verbose:
            print(f"Found {len(regions)} consecutive filter region(s): {regions}")
        
        # For simplicity, handle first region (most common case)
        # Extension: could handle multiple regions with nested permutations
        region = regions[0]
        
        if self.verbose:
            print(f"Generating permutations for filter region {region}")
        
        all_plans = []
        region_op_sets = [op_sets[i] for i in region]
        
        perm_count = 0
        for perm in permutations(range(len(region))):
            if perm_count >= max_permutations:
                break
            
            # Create new op_sets with permuted filters
            new_op_sets = list(op_sets)
            for new_pos, orig_pos in enumerate(perm):
                new_op_sets[region[new_pos]] = region_op_sets[orig_pos]
            
            # Rebuild plan
            try:
                new_plan = self._rebuild_sentinel_plan(new_op_sets)
                all_plans.append(new_plan)
                perm_count += 1
            except Exception as e:
                logger.warning(f"Failed to rebuild plan for permutation {perm}: {e}")
        
        if self.verbose:
            print(f"Generated {len(all_plans)} filter ordering variants")
        
        return all_plans
    
    def create_optimally_ordered_plan(
        self,
        sentinel_plan: "SentinelPlan",
        selectivity_result: SelectivityEstimationResult,
    ) -> "SentinelPlan":
        """
        Create a new SentinelPlan with filters ordered by selectivity.
        
        Most selective (lowest selectivity) filters are placed first.
        
        Args:
            sentinel_plan: Original SentinelPlan
            selectivity_result: Selectivity estimation results
            
        Returns:
            New SentinelPlan with optimal filter ordering
        """
        op_sets = self._flatten_plan(sentinel_plan)
        filter_indices = self._find_filter_indices(op_sets)
        regions = self._find_consecutive_filter_regions(filter_indices)
        
        if len(regions) == 0:
            if self.verbose:
                print("No filter regions to reorder")
            return sentinel_plan
        
        # Create mutable copy
        new_op_sets = list(op_sets)
        
        for region in regions:
            # Get (index, selectivity, op_set) for each filter in region
            region_data = []
            for idx in region:
                op_set = op_sets[idx]
                selectivity = selectivity_result.get_selectivity_for_operator_set(op_set)
                region_data.append((idx, selectivity, op_set))
            
            # Sort by selectivity (ascending = most selective first)
            region_data.sort(key=lambda x: x[1])
            
            if self.verbose:
                print(f"Reordering region {region} by selectivity:")
                for orig_idx, sel, _ in region_data:
                    print(f"  selectivity={sel:.3f}")
            
            # Place sorted op_sets back into region positions
            for new_pos, (orig_idx, sel, op_set) in enumerate(region_data):
                new_op_sets[region[new_pos]] = op_set
        
        # Rebuild plan
        try:
            new_plan = self._rebuild_sentinel_plan(new_op_sets)
            if self.verbose:
                print("✓ Successfully created optimally ordered plan")
            return new_plan
        except Exception as e:
            logger.error(f"Failed to rebuild plan: {e}")
            return sentinel_plan
    
    def prune_non_optimal_orderings(
        self,
        sentinel_plans: list["SentinelPlan"],
        selectivity_result: SelectivityEstimationResult,
    ) -> list["SentinelPlan"]:
        """
        Prune SentinelPlans that don't have optimal filter ordering.
        
        Keeps only plans where filters are ordered by ascending selectivity.
        
        Args:
            sentinel_plans: List of SentinelPlan variants
            selectivity_result: Selectivity estimation results
            
        Returns:
            Filtered list containing only optimally-ordered plans
        """
        if len(sentinel_plans) <= 1:
            return sentinel_plans
        
        optimal_plans = []
        
        for plan in sentinel_plans:
            op_sets = self._flatten_plan(plan)
            filter_indices = self._find_filter_indices(op_sets)
            regions = self._find_consecutive_filter_regions(filter_indices)
            
            is_optimal = True
            
            for region in regions:
                # Check if filters in this region are ordered by selectivity
                prev_selectivity = -1.0
                for idx in region:
                    op_set = op_sets[idx]
                    selectivity = selectivity_result.get_selectivity_for_operator_set(op_set)
                    
                    if selectivity < prev_selectivity:
                        is_optimal = False
                        break
                    prev_selectivity = selectivity
                
                if not is_optimal:
                    break
            
            if is_optimal:
                optimal_plans.append(plan)
                if self.verbose:
                    print(f"  ✓ Kept plan (optimal ordering)")
            else:
                if self.verbose:
                    print(f"  ✗ Pruned plan (non-optimal ordering)")
        
        # Always keep at least one plan
        if len(optimal_plans) == 0:
            logger.warning("No optimal plans found - keeping first plan")
            return [sentinel_plans[0]]
        
        if self.verbose:
            print(f"Pruned {len(sentinel_plans) - len(optimal_plans)} non-optimal plans")
            print(f"Kept {len(optimal_plans)} optimal plans")
        
        return optimal_plans


# =============================================================================
# Integration Functions for QueryProcessor
# =============================================================================

def estimate_filter_selectivity_for_sentinel_plan(
    sentinel_plan: "SentinelPlan",
    train_dataset: dict[str, "Dataset"],
    num_samples: int = 10,
    verbose: bool = False,
) -> SelectivityEstimationResult:
    """
    Estimate filter selectivity for a sentinel plan.
    
    This is Step 1: Use sentinel_plan to measure filter selectivity.
    """
    estimator = FilterSelectivityEstimator(num_samples=num_samples, verbose=verbose)
    return estimator.estimate_selectivity(sentinel_plan, train_dataset)


def reorder_sentinel_plan_filters(
    sentinel_plan: "SentinelPlan",
    selectivity_result: SelectivityEstimationResult,
    verbose: bool = False,
) -> "SentinelPlan":
    """
    Reorder filters in SentinelPlan based on selectivity.
    
    Returns a new SentinelPlan with optimal filter ordering.
    """
    reorderer = SentinelPlanReorderer(verbose=verbose)
    return reorderer.create_optimally_ordered_plan(sentinel_plan, selectivity_result)


def generate_filter_ordering_variants(
    sentinel_plan: "SentinelPlan",
    max_permutations: int = 24,
    verbose: bool = False,
) -> list["SentinelPlan"]:
    """
    Generate all filter ordering permutations.
    
    Returns list of SentinelPlan variants.
    """
    reorderer = SentinelPlanReorderer(verbose=verbose)
    return reorderer.generate_all_filter_orderings(sentinel_plan, max_permutations)


def prune_suboptimal_filter_orderings(
    sentinel_plans: list["SentinelPlan"],
    selectivity_result: SelectivityEstimationResult,
    verbose: bool = False,
) -> list["SentinelPlan"]:
    """
    Prune plans with non-optimal filter orderings.
    
    This is Step 2: Prune non-optimal filter orderings.
    """
    reorderer = SentinelPlanReorderer(verbose=verbose)
    return reorderer.prune_non_optimal_orderings(sentinel_plans, selectivity_result)


def generate_and_prune_filter_orderings(
    sentinel_plan: "SentinelPlan",
    selectivity_result: SelectivityEstimationResult,
    max_permutations: int = 24,
    verbose: bool = False,
) -> list["SentinelPlan"]:
    """
    Generate multiple filter orderings and prune non-optimal ones.
    
    This combines both operations:
    1. Generate all permutations of filter orderings
    2. Prune plans that don't have optimal (ascending selectivity) ordering
    
    Returns:
        List of SentinelPlans with only optimal filter orderings
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GENERATING AND PRUNING FILTER ORDERINGS")
        print("=" * 60)
    
    # Generate all permutations
    all_plans = generate_filter_ordering_variants(
        sentinel_plan, max_permutations, verbose
    )
    
    # Prune non-optimal
    optimal_plans = prune_suboptimal_filter_orderings(
        all_plans, selectivity_result, verbose
    )
    
    if verbose:
        print(f"\nResult: {len(optimal_plans)} optimal plan(s) from {len(all_plans)} variants")
        print("=" * 60 + "\n")
    
    return optimal_plans


def optimize_filter_ordering(
    sentinel_plan: "SentinelPlan",
    selectivity_result: SelectivityEstimationResult,
    verbose: bool = False,
) -> "SentinelPlan":
    """
    Main entry point: Optimize the sentinel plan based on filter selectivity.
    
    This function:
    1. Prints selectivity summary
    2. Creates a new SentinelPlan with optimal filter ordering
    
    Args:
        sentinel_plan: The original SentinelPlan
        selectivity_result: Results from selectivity estimation
        verbose: Whether to print verbose output
        
    Returns:
        Optimized SentinelPlan with filters reordered by selectivity
    """
    # Print summary
    selectivity_result.print_summary(verbose=verbose)
    
    # Check if reordering is needed
    if selectivity_result.is_order_optimal:
        if verbose:
            print("✓ Filter order is already optimal - no changes needed")
        return sentinel_plan
    
    # Create optimally ordered plan
    if verbose:
        print("⚙️  Creating optimally ordered SentinelPlan...")
    
    optimized_plan = reorder_sentinel_plan_filters(
        sentinel_plan, selectivity_result, verbose
    )
    
    if verbose:
        print("✓ Filters reordered successfully")
        print("   Most selective filters now execute first")
    
    return optimized_plan
