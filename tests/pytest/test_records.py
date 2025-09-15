from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from palimpzest.core.elements.records import DataRecord


# Example test schema
class TestSchema(BaseModel):
    name: str = Field(description="Test name field")
    value: Any = Field(description="Test value field")


class TestDataRecord:
    @pytest.fixture
    def sample_record(self):
        """Fixture to create a sample DataRecord for testing"""
        record = DataRecord(data_item=TestSchema(name="test", value=42), source_indices=[0])
        return record

    @pytest.fixture
    def sample_df(self):
        """Fixture to create a sample DataFrame for testing"""
        return pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'value': [1, 2]
        })

    def test_create_record(self, sample_record):
        """Test basic record creation and attribute access"""
        assert sample_record.name == "test"
        assert sample_record.value == 42
        assert sample_record._source_indices == [0]

    def test_record_equality(self, sample_record):
        """Test record equality comparison"""
        record2 = DataRecord(data_item=TestSchema(name="test", value=42), source_indices=[0])
        assert sample_record == record2

    def test_to_df(self, sample_df):
        """Test converting records back to DataFrame"""
        records = [
            DataRecord(data_item=TestSchema(name="Alice", value=1), source_indices=[0]),
            DataRecord(data_item=TestSchema(name="Bob", value=2), source_indices=[1]),
        ]
        df_result = DataRecord.to_df(records)
        assert df_result.equals(sample_df)

    def test_to_df_with_project_cols(self, sample_df):
        """Test converting records to DataFrame with project_cols"""
        records = [
            DataRecord(data_item=TestSchema(name="Alice", value=1), source_indices=[0]),
            DataRecord(data_item=TestSchema(name="Bob", value=2), source_indices=[1]),
        ]
        df_result = DataRecord.to_df(records, project_cols=["name"])
        assert df_result.equals(sample_df[["name"]])

    def test_invalid_attribute(self, sample_record):
        """Test accessing non-existent attribute"""
        with pytest.raises(AttributeError):
            _ = sample_record.nonexistent_field

    def test_to_dict(self, sample_record):
        """Test dictionary representation"""
        record_dict = sample_record.to_dict()
        assert record_dict['name'] == 'test'
        assert record_dict['value'] == 42

    def test_to_json_str(self, sample_record):
        """Test JSON string representation"""
        json_str = sample_record.to_json_str()
        assert 'test' in json_str
        assert '42' in json_str


if __name__ == '__main__':
    pytest.main([__file__])