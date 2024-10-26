from palimpzest.operators.aggregate import AggregateOp, ApplyGroupByOp, AverageAggregateOp, CountAggregateOp
from palimpzest.operators.convert import ConvertOp, LLMConvert, LLMConvertBonded, LLMConvertConventional, NonLLMConvert
from palimpzest.operators.datasource import CacheScanDataOp, DataSourcePhysicalOp, MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp, LLMFilter, NonLLMFilter
from palimpzest.operators.limit import LimitScanOp
from palimpzest.operators.logical import (
    Aggregate,
    BaseScan,
    CacheScan,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
    LogicalOperator,
)
from palimpzest.operators.physical import PhysicalOperator

LOGICAL_OPERATORS = [
    LogicalOperator,
    Aggregate,
    BaseScan,
    CacheScan,
    ConvertScan,
    FilteredScan,
    GroupByAggregate,
    LimitScan,
]


PHYSICAL_OPERATORS = (
    # aggregate
    [AggregateOp, ApplyGroupByOp, AverageAggregateOp, CountAggregateOp]
    # convert
    + [ConvertOp, NonLLMConvert, LLMConvert, LLMConvertConventional, LLMConvertBonded]
    # datasource
    + [DataSourcePhysicalOp, MarshalAndScanDataOp, CacheScanDataOp]
    # filter
    + [FilterOp, NonLLMFilter, LLMFilter]
    # limit
    + [LimitScanOp]
    # physical
    + [PhysicalOperator]
)


__all__ = [
    "LOGICAL_OPERATORS",
    "PHYSICAL_OPERATORS",
]
