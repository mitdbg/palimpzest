from palimpzest.core.data.datareaders import MemorySource
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import Schema
from palimpzest.query.operators.datasource import MarshalAndScanDataOp


class List(Schema):
    value = Field(desc="List item")


def test_marshal_and_scan_memory_source():
    # Create test data
    test_data = ["test1", "test2", "test3"]
    
    # Create MemorySource with test data
    memory_source = MemorySource(test_data, dataset_id="test_dataset")
    
    # Create MarshalAndScanDataOp
    op = MarshalAndScanDataOp(output_schema=List, datasource=memory_source)
    
    # Execute the scan operator on the first source record
    result = op(0)
    
    assert len(result.data_records) == 1
    assert result.data_records[0].value == "test1"
    
    # Test stats
    assert len(result.record_op_stats) == 1
    stats = result.record_op_stats[0]
    assert stats.op_name == "MarshalAndScanDataOp"
    assert stats.op_details["output_schema"] == str(List)
    assert stats.time_per_record > 0
    assert stats.cost_per_record == 0.0

# def test_marshal_and_scan_memory_source_multiple_records():
#     # Test with numeric data
#     test_data = [1, 2, 3, 4, 5]
#     memory_source = MemorySource(test_data, schema=List, dataset_id="test_numbers")

#     op = MarshalAndScanDataOp(datasource=memory_source)

#     # Test each index
#     for idx in range(len(memory_source)):
#         result = op(idx)

#         # Verify results
#         assert len(result.records) == 1
#         assert result.records[0].value == test_data[idx]
#         assert len(result.record_op_stats) == 1

# def test_marshal_and_scan_empty_source():
#     # Test with empty data
#     memory_source = MemorySource([], schema=List, dataset_id="empty_dataset")

#     op = MarshalAndScanDataOp(datasource=memory_source)

#     # Should raise IndexError when trying to access empty source
#     with pytest.raises(IndexError):
#         op(0)
