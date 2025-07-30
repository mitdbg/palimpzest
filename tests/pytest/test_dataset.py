import pandas as pd
import pytest

from palimpzest.core.data.dataset import Dataset
from palimpzest.core.data.iter_dataset import MemoryDataset
from palimpzest.query.operators.logical import ConvertScan, FilteredScan


# Test data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })


def test_dataset_initialization(sample_df):
    ds = MemoryDataset("test", sample_df)
    assert isinstance(ds, Dataset)
    assert sorted(ds.schema.model_fields) == ['age', 'id', 'name']


def test_dataset_filter(sample_df):
    ds = MemoryDataset("test", sample_df)
    
    # Test callable filter
    filtered_ds = ds.filter(lambda x: x['age'] > 30)
    assert isinstance(filtered_ds, Dataset)
    assert isinstance(filtered_ds._operator, FilteredScan)
    
    # Test semantic filter
    sem_filtered_ds = ds.sem_filter("age > 30")
    assert isinstance(sem_filtered_ds, Dataset)
    assert isinstance(filtered_ds._operator, FilteredScan)


def test_dataset_add_columns(sample_df):
    ds = MemoryDataset("test", sample_df)

    # Test UDF add_columns
    def add_greeting(df):
        df['greeting'] = 'Hello ' + df['name']
        return df
    
    new_ds = ds.add_columns(udf=add_greeting, cols=[{'name': 'greeting', 'desc': 'Greeting message', 'type': str}])
    assert isinstance(new_ds, Dataset)
    assert isinstance(new_ds._operator, ConvertScan)
    assert new_ds._operator.udf is not None
    assert sorted(new_ds.schema.model_fields) == ['age', 'greeting', 'id', 'name']
    greeting_field = new_ds.schema.model_fields['greeting'] 
    assert greeting_field.annotation is str
    assert greeting_field.description == 'Greeting message'

    # Test semantic add_columns
    new_cols = [{'name': 'greeting', 'type': str, 'desc': 'Greeting message'},
                {'name': 'score', 'type': int | float, 'desc': 'Score'}]
    sem_new_ds = ds.sem_add_columns(new_cols)
    assert isinstance(sem_new_ds, Dataset)
    assert isinstance(sem_new_ds._operator, ConvertScan)
    assert sorted(sem_new_ds.schema.model_fields) == ['age', 'greeting', 'id', 'name', 'score']
    greeting_field = sem_new_ds.schema.model_fields['greeting']
    assert greeting_field.annotation is str
    assert greeting_field.description == 'Greeting message'

    score_field = sem_new_ds.schema.model_fields['score']
    assert score_field.annotation == int | float
    assert score_field.description == 'Score'

    with pytest.raises(ValueError, match="`udf` and `cols` must be provided for add_columns."):
        ds.add_columns(udf=add_greeting, cols=None)
