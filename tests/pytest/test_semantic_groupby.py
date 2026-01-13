#!/usr/bin/env python3
"""
Test script for semantic group by operation.

This script tests the SemanticGroupByOp implementation by creating a small dataset
of product reviews and grouping them by complaint type.
"""

import pandas as pd

import palimpzest as pz
from palimpzest.constants import Model
from palimpzest.query.operators.aggregate import SemanticGroupByOp

# Define columns for the review schema
review_cols = [
    {"name": "complaint", "type": str, "desc": "The type of complaint mentioned in the review (e.g., size, quality, shipping, description mismatch, ergonomics)"},
]

def test_semantic_groupby_basic():
    """Test basic semantic group by functionality using the physical operator directly."""
    # Create list of candidates from text file dataset with schema
    ds = pz.TextFileDataset(id="reviews", path="tests/pytest/data/product-reviews/")
    output = ds.run()
    candidates = [dr for dr in output]
    
    print(f"Loaded {len(candidates)} review candidates with schema")
    print(f"Sample candidate fields: {list(candidates[0].to_dict().keys()) if candidates else 'none'}")
    
    # Get input schema from the candidates
    input_schema = candidates[0].schema if candidates else None
    
    # Create output schema (group by field + count)
    # Using the same naming convention as Dataset.sem_groupby()
    from typing import Any

    from palimpzest.core.lib.schemas import create_schema_from_fields

    # define the groupby and aggregate fields
    gby_fields = ['complaint']
    agg_fields = ['contents']
    agg_funcs = ['count']
    
    fields = []
    # Add group by fields to output schema
    for g in gby_fields:
        f = {"name": g, "type": Any, "desc": f"Group by field: {g}"}
        fields.append(f)
    
    # Add aggregation fields to output schema
    for agg_field_name in agg_fields:
        f = {"name": agg_field_name, "type": Any, "desc": f"Aggregate field: {agg_field_name}"}
        fields.append(f)

    output_schema = create_schema_from_fields(fields)

    # Create instance of the physical operator
    sem_group_by_op = SemanticGroupByOp(
        gby_fields=gby_fields,
        agg_fields=agg_fields,
        agg_funcs=agg_funcs,
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
    
    assert False

def test_semantic_groupby_via_dataset():
    """Test semantic group by via Dataset API."""
    # Create dataset and add schema
    ds = pz.TextFileDataset(id="reviews", path="tests/pytest/data/product-reviews/")
    
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
