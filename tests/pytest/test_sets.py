import pandas as pd
import pytest

from palimpzest.core.lib.schemas import DefaultSchema, NumericField, StringField
from palimpzest.sets import Dataset


# Test data
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35]
    })


def test_dataset_initialization(sample_df):
    ds = Dataset(sample_df)
    assert isinstance(ds, Dataset)
    assert ds.schema.field_names() == ['age', 'id', 'name']

    # Test with default schema
    ds = Dataset(sample_df, schema=DefaultSchema)
    assert ds.schema == DefaultSchema

def test_dataset_filter(sample_df):
    ds = Dataset(sample_df)
    
    # Test callable filter
    filtered_ds = ds.filter(lambda x: x['age'] > 30)
    assert isinstance(filtered_ds, Dataset)
    assert filtered_ds._filter is not None
    
    # Test semantic filter
    sem_filtered_ds = ds.sem_filter("age > 30")
    assert isinstance(sem_filtered_ds, Dataset)
    assert sem_filtered_ds._filter is not None


def test_dataset_add_columns(sample_df):
    ds = Dataset(sample_df)

    # Test UDF add_columns
    def add_greeting(df):
        df['greeting'] = 'Hello ' + df['name']
        return df
    
    new_ds = ds.add_columns(udf=add_greeting, types=[{'name': 'greeting', 'type': 'string'}])
    assert isinstance(new_ds, Dataset)
    assert new_ds._udf is not None
    assert new_ds.schema.field_names() == ['age', 'greeting', 'id', 'name']
    greeting_field = new_ds.schema.field_map()['greeting'] 
    assert isinstance(greeting_field, StringField)
    assert greeting_field.desc == 'New column: greeting'

    # Test semantic add_columns
    new_cols = [{'name': 'greeting', 'type': 'string', 'desc': 'Greeting message'},
                {'name': 'score', 'type': 'numeric', 'desc': 'Score'}]
    sem_new_ds = ds.sem_add_columns(new_cols)
    assert isinstance(sem_new_ds, Dataset)
    assert sem_new_ds.schema.field_names() == ['age', 'greeting', 'id', 'name', 'score']
    greeting_field = sem_new_ds.schema.field_map()['greeting']
    assert isinstance(greeting_field, StringField)
    assert greeting_field.desc == 'Greeting message'


    score_field = sem_new_ds.schema.field_map()['score']
    assert isinstance(score_field, NumericField)
    assert score_field.desc == 'Score'

    with pytest.raises(ValueError, match="udf and types must be provided for add_columns."):
        ds.add_columns(udf=add_greeting)


