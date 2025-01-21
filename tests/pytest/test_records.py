import pandas as pd
import pytest
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import Schema


# Example test schema
class TestSchema(Schema):
    name = Field(desc="Test name field")
    value = Field(desc="Test value field")


class TestDataRecord:
    @pytest.fixture
    def sample_record(self):
        """Fixture to create a sample DataRecord for testing"""
        record = DataRecord(schema=TestSchema, source_id="test_source")
        record.name = "test"
        record.value = 42
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
        assert sample_record.source_id == "test_source"

    def test_record_equality(self, sample_record):
        """Test record equality comparison"""
        record2 = DataRecord(schema=TestSchema, source_id="test_source")
        record2.name = "test"
        record2.value = 42
        assert sample_record == record2

    def test_from_df(self, sample_df):
        """Test creating records from DataFrame"""
        records = DataRecord.from_df(sample_df, schema=TestSchema)
        assert len(records) == 2
        assert records[0].name == "Alice"
        assert records[1].value == 2

    def test_as_df(self, sample_df):
        """Test converting records back to DataFrame"""
        records = DataRecord.from_df(sample_df, schema=TestSchema)
        df_result = DataRecord.as_df(records)
        pd.testing.assert_frame_equal(df_result, sample_df)

    def test_derived_schema(self, sample_df):
        """Test auto-schema generation from DataFrame"""
        records = DataRecord.from_df(sample_df)  # No schema provided
        assert len(records) == 2
        assert hasattr(records[0].schema, 'name')
        assert hasattr(records[0].schema, 'value')

    def test_invalid_attribute(self, sample_record):
        """Test accessing non-existent attribute"""
        with pytest.raises(AttributeError):
            _ = sample_record.nonexistent_field

    def test_as_dict(self, sample_record):
        """Test dictionary representation"""
        record_dict = sample_record.as_dict()
        assert record_dict['name'] == 'test'
        assert record_dict['value'] == 42

    def test_as_json_str(self, sample_record):
        """Test JSON string representation"""
        json_str = sample_record.as_json_str()
        assert 'test' in json_str
        assert '42' in json_str


if __name__ == '__main__':
    pytest.main([__file__])