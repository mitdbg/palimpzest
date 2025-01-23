import pytest
from palimpzest.core.data.datasources import MemorySource
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import Schema, List
from palimpzest.query.operators.datasource import MarshalAndScanDataOp
from palimpzest.core.lib.schemas import SourceRecord

def test_marshal_and_scan_memory_source():
    # Create test data
    test_data = ["test1", "test2", "test3"]
    
    # Create MemorySource with test data
    memory_source = MemorySource(test_data, dataset_id="test_dataset")
    
    # Create MarshalAndScanDataOp
    op = MarshalAndScanDataOp(output_schema=List, dataset_id="test_dataset")

    current_scan_idx = 0
    candidate = DataRecord(schema=SourceRecord, source_id=current_scan_idx)
    candidate.idx = current_scan_idx
    candidate.get_item_fn = memory_source.get_item
    
    # Execute the operator
    result = op(candidate)
    
    assert len(result.data_records) == 1
    assert result.data_records[0].value == "test1"
    
    # Test stats
    assert len(result.record_op_stats) == 1
    stats = result.record_op_stats[0]
    assert stats.op_name == "MarshalAndScanDataOp"
    assert stats.op_details["dataset_id"] == "test_dataset"
    assert stats.time_per_record > 0
    assert stats.cost_per_record == 0.0

# def test_marshal_and_scan_memory_source_multiple_records():
#     # Test with numeric data
#     test_data = [1, 2, 3, 4, 5]
#     memory_source = MemorySource(test_data, schema=List, dataset_id="test_numbers")
    
#     op = MarshalAndScanDataOp(dataset_id="test_numbers")
    
#     # Test each index
#     for idx in range(len(test_data)):
#         mock_record = DataRecord(Schema())
#         mock_record.idx = idx
#         mock_record.get_item_fn = memory_source.get_item
        
#         result = op(mock_record)
        
#         # Verify results
#         assert len(result.records) == 1
#         assert result.records[0].value == test_data[idx]
#         assert len(result.record_op_stats) == 1

# def test_marshal_and_scan_empty_source():
#     # Test with empty data
#     memory_source = MemorySource([], schema=List, dataset_id="empty_dataset")
    
#     op = MarshalAndScanDataOp(dataset_id="empty_dataset")
    
#     mock_record = DataRecord(Schema())
#     mock_record.idx = 0
#     mock_record.get_item_fn = memory_source.get_item
    
#     # Should raise IndexError when trying to access empty source
#     with pytest.raises(IndexError):
#         op(mock_record)




