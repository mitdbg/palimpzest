from palimpzest.operators.aggregate import AggregateOp as _AggregateOp
from palimpzest.operators.aggregate import ApplyGroupByOp as _ApplyGroupByOp
from palimpzest.operators.aggregate import AverageAggregateOp as _AverageAggregateOp
from palimpzest.operators.aggregate import CountAggregateOp as _CountAggregateOp
from palimpzest.operators.convert import ConvertOp as _ConvertOp
from palimpzest.operators.convert import LLMConvert as _LLMConvert
from palimpzest.operators.convert import LLMConvertBonded as _LLMConvertBonded
from palimpzest.operators.convert import LLMConvertConventional as _LLMConvertConventional
from palimpzest.operators.convert import NonLLMConvert as _NonLLMConvert
from palimpzest.operators.datasource import CacheScanDataOp as _CacheScanDataOp
from palimpzest.operators.datasource import DataSourcePhysicalOp as _DataSourcePhysicalOp
from palimpzest.operators.datasource import MarshalAndScanDataOp as _MarshalAndScanDataOp
from palimpzest.operators.filter import FilterOp as _FilterOp
from palimpzest.operators.filter import LLMFilter as _LLMFilter
from palimpzest.operators.filter import NonLLMFilter as _NonLLMFilter
from palimpzest.operators.limit import LimitScanOp as _LimitScanOp
from palimpzest.operators.logical import (
    Aggregate as _Aggregate,
)
from palimpzest.operators.logical import (
    BaseScan as _BaseScan,
)
from palimpzest.operators.logical import (
    CacheScan as _CacheScan,
)
from palimpzest.operators.logical import (
    ConvertScan as _ConvertScan,
)
from palimpzest.operators.logical import (
    FilteredScan as _FilteredScan,
)
from palimpzest.operators.logical import (
    GroupByAggregate as _GroupByAggregate,
)
from palimpzest.operators.logical import (
    LimitScan as _LimitScan,
)
from palimpzest.operators.logical import (
    LogicalOperator as _LogicalOperator,
)
from palimpzest.operators.physical import PhysicalOperator as _PhysicalOperator

LOGICAL_OPERATORS = [
    _LogicalOperator,
    _Aggregate,
    _BaseScan,
    _CacheScan,
    _ConvertScan,
    _FilteredScan,
    _GroupByAggregate,
    _LimitScan,
]

PHYSICAL_OPERATORS = (
    # aggregate
    [_AggregateOp, _ApplyGroupByOp, _AverageAggregateOp, _CountAggregateOp]
    # convert
    + [_ConvertOp, _NonLLMConvert, _LLMConvert, _LLMConvertConventional, _LLMConvertBonded]
    # datasource
    + [_DataSourcePhysicalOp, _MarshalAndScanDataOp, _CacheScanDataOp]
    # filter
    + [_FilterOp, _NonLLMFilter, _LLMFilter]
    # limit
    + [_LimitScanOp]
    # physical
    + [_PhysicalOperator]
)

__all__ = [
    "LOGICAL_OPERATORS",
    "PHYSICAL_OPERATORS",
]
