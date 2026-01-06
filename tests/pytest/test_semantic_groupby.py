#!/usr/bin/env python3
"""
Test script for semantic group by operation.

This script tests the SemanticGroupByOp implementation by creating a small dataset
of product reviews and grouping them by complaint type.
"""

import pandas as pd
import palimpzest as pz
from palimpzest.query.operators.aggregate import SemanticGroupByOp
from palimpzest.constants import Model

# Define columns for the review schema
review_cols = [
    {"name": "complaint", "type": str, "desc": "The type of complaint mentioned in the review (e.g., size, quality, shipping, description mismatch, ergonomics)"},
]

def test_semantic_groupby_basic():
    """Test basic semantic group by functionality using the physical operator directly."""
    print("Testing SemanticGroupByOp basic functionality...")
    
    try:
        # Create list of candidates from text file dataset with schema
        ds = pz.TextFileDataset(id="reviews", path="product-reviews/")
        output = ds.run()
        candidates = [dr for dr in output]
        
        print(f"Loaded {len(candidates)} review candidates with schema")
        print(f"Sample candidate fields: {list(candidates[0].to_dict().keys()) if candidates else 'none'}")
        
        # Get input schema from the candidates
        input_schema = candidates[0].schema if candidates else None
        
        # Create output schema (group by field + count)
        # Using the same naming convention as Dataset.sem_groupby()
        from palimpzest.core.lib.schemas import create_schema_from_fields
        from typing import Any
        
        fields = []
        # Add group by fields to output schema
        for g in ['complaint']:
            f = {"name": g, "type": Any, "desc": f"Group by field: {g}"}
            fields.append(f)
        
        # Add aggregation fields to output schema
        agg_fields_list = ['contents']
        agg_funcs_list = ['count']
        for i, agg_func in enumerate(agg_funcs_list):
            agg_field_name = f"{agg_func}({agg_fields_list[i]})"
            f = {"name": agg_field_name, "type": Any, "desc": f"Aggregate field: {agg_field_name}"}
            fields.append(f)
        
        output_schema = create_schema_from_fields(fields)
        
        # Create instance of the physical operator
        sem_group_by_op = SemanticGroupByOp(
            gby_fields=['complaint'], 
            agg_fields=['contents'], 
            agg_funcs=['count'],
            input_schema=input_schema,
            output_schema=output_schema,
            model=Model.GPT_4o_MINI,
            logical_op_id="test_semantic_groupby",  # Required for RecordOpStats
            verbose=False
        )
        
        print(f"Created SemanticGroupByOp: {sem_group_by_op}")
        
        # Execute the group by operation
        grouped_output = sem_group_by_op(candidates)
        
        # Convert to DataFrame and print
        df = pd.DataFrame([dr.to_dict() for dr in grouped_output])
        print("\nGrouped Results:")
        print(df)
        print(f"\nTotal groups: {len(df)}")
        # print(f"Total cost: ${grouped_output.stats.cost:.4f}")
        # print(f"Total time: {grouped_output.stats.time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_groupby_via_dataset():
    """Test semantic group by via Dataset API."""
    print("\nTesting sem_groupby via Dataset API...")
    
    try:
        # Create dataset and add schema
        ds = pz.TextFileDataset(id="reviews", path="product-reviews/")
        
        # Apply semantic group by operation
        ds = ds.sem_groupby(
            gby_fields=['complaint'], 
            agg_fields=['contents'], 
            agg_funcs=['count']
        )
        
        # Run the query
        output = ds.run()
        
        # Convert to DataFrame and print
        df = output.to_df()
        print("\nGrouped Results:")
        print(df)
        print(f"\nTotal groups: {len(df)}")
        # print(f"Total cost: ${output.stats.cost:.4f}")
        # print(f"Total time: {output.stats.time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Semantic GroupBy Test Suite")
    print("=" * 80)
    
    print("\nRunning tests...\n")
    
    # Run tests
    print("Test 1: Basic SemanticGroupByOp")
    test_semantic_groupby_basic()
    
    print("\n" + "=" * 80)
    print("Test 2: Dataset.sem_groupby() API")
    test_semantic_groupby_via_dataset()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
