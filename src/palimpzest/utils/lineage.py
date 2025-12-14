from typing import Any, List, Dict

def get_lineage_exploder_udf(edge_type: str = "HAS_MEMBER") -> callable:
    """
    Returns a UDF that explodes a record into a list of (src, dst) pairs based on its lineage.
    
    NOTE: This relies on the execution engine passing the full DataRecord object to the UDF
    if the UDF signature requests it, OR we need to modify the execution engine.
    
    Currently, PZ passes `candidate.to_dict()` to UDFs.
    However, `DataRecord.to_dict()` does NOT include `_parent_ids`.
    
    Workaround: We will implement a new PhysicalOperator `ExplodeLineage` that handles this.
    """
    pass
