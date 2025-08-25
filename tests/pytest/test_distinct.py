"""This testing class is an integration test suite.
What it does is consider one of the demo scenarios and test whether we can obtain the same results with the refactored code
"""

import os

import pandas as pd
import pytest

import palimpzest as pz

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


# Test data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'person_id': [1, 1, 2, 3, 4],
        'name': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie'],
        'age': [25, 25, 30, 30, 35]
    })


@pytest.mark.parametrize("execution_strategy", ["sequential", "parallel"])
def test_distinct(sample_df, execution_strategy):
    ds = pz.MemoryDataset("test", sample_df)
    ds = ds.distinct()
    output = ds.run(config=pz.QueryProcessorConfig(execution_strategy=execution_strategy))
    output_df = output.to_df()
    assert len(output_df) == 4
    assert sorted(output_df.columns) == ['age', 'name', 'person_id']


@pytest.mark.parametrize("execution_strategy", ["sequential", "parallel"])
def test_dataset_with_distinct_cols(sample_df, execution_strategy):
    ds = pz.MemoryDataset("test", sample_df)
    ds = ds.distinct(distinct_cols=['name', 'age'])
    output = ds.run(config=pz.QueryProcessorConfig(execution_strategy=execution_strategy))
    output_df = output.to_df()
    assert len(output_df) == 3
    assert sorted(output_df.columns) == ['age', 'name', 'person_id']


@pytest.mark.parametrize("execution_strategy", ["sequential", "parallel"])
def test_dataset_with_distinct_cols_and_limit(sample_df, execution_strategy):
    ds = pz.MemoryDataset("test", sample_df)
    ds = ds.distinct(distinct_cols=['name', 'age']).limit(2)
    output = ds.run(config=pz.QueryProcessorConfig(execution_strategy=execution_strategy))
    output_df = output.to_df()
    assert len(output_df) == 2
    assert sorted(output_df.columns) == ['age', 'name', 'person_id']


@pytest.mark.parametrize("execution_strategy", ["sequential", "parallel"])
def test_dataset_with_distinct_cols_and_filter(sample_df, execution_strategy):
    ds = pz.MemoryDataset("test", sample_df)
    ds = ds.distinct(distinct_cols=['name', 'age']).filter(lambda row: row['age'] > 30)
    output = ds.run(config=pz.QueryProcessorConfig(execution_strategy=execution_strategy))
    output_df = output.to_df()
    assert len(output_df) == 1
    assert sorted(output_df.columns) == ['age', 'name', 'person_id']
