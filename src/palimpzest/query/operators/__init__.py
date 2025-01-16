from palimpzest.query.operators.aggregate import AggregateOp as _AggregateOp
from palimpzest.query.operators.aggregate import ApplyGroupByOp as _ApplyGroupByOp
from palimpzest.query.operators.aggregate import AverageAggregateOp as _AverageAggregateOp
from palimpzest.query.operators.aggregate import CountAggregateOp as _CountAggregateOp
from palimpzest.query.operators.convert import ConvertOp as _ConvertOp
from palimpzest.query.operators.convert import LLMConvert as _LLMConvert
from palimpzest.query.operators.convert import LLMConvertBonded as _LLMConvertBonded
from palimpzest.query.operators.convert import LLMConvertConventional as _LLMConvertConventional
from palimpzest.query.operators.convert import NonLLMConvert as _NonLLMConvert
from palimpzest.query.operators.datasource import CacheScanDataOp as _CacheScanDataOp
from palimpzest.query.operators.datasource import DataSourcePhysicalOp as _DataSourcePhysicalOp
from palimpzest.query.operators.datasource import MarshalAndScanDataOp as _MarshalAndScanDataOp
from palimpzest.query.operators.filter import FilterOp as _FilterOp
from palimpzest.query.operators.filter import LLMFilter as _LLMFilter
from palimpzest.query.operators.filter import NonLLMFilter as _NonLLMFilter
from palimpzest.query.operators.limit import LimitScanOp as _LimitScanOp
from palimpzest.query.operators.logical import (
    Aggregate as _Aggregate,
)
from palimpzest.query.operators.logical import (
    BaseScan as _BaseScan,
)
from palimpzest.query.operators.logical import (
    CacheScan as _CacheScan,
)
from palimpzest.query.operators.logical import (
    ConvertScan as _ConvertScan,
)
from palimpzest.query.operators.logical import (
    FilteredScan as _FilteredScan,
)
from palimpzest.query.operators.logical import (
    GroupByAggregate as _GroupByAggregate,
)
from palimpzest.query.operators.logical import (
    LimitScan as _LimitScan,
)
from palimpzest.query.operators.logical import (
    LogicalOperator as _LogicalOperator,
)
from palimpzest.query.operators.logical import (
    Project as _Project,
)
from palimpzest.query.operators.logical import (
    RetrieveScan as _RetrieveScan,
)
from palimpzest.query.operators.mixture_of_agents_convert import MixtureOfAgentsConvert as _MixtureOfAgentsConvert
from palimpzest.query.operators.physical import PhysicalOperator as _PhysicalOperator
from palimpzest.query.operators.project import ProjectOp as _ProjectOp
from palimpzest.query.operators.retrieve import RetrieveOp as _RetrieveOp

LOGICAL_OPERATORS = [
    _LogicalOperator,
    _Aggregate,
    _BaseScan,
    _CacheScan,
    _ConvertScan,
    _FilteredScan,
    _GroupByAggregate,
    _LimitScan,
    _Project,
    _RetrieveScan,
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
    # mixture-of-agents
    + [_MixtureOfAgentsConvert]
    # physical
    + [_PhysicalOperator]
    # project
    + [_ProjectOp]
    # retrieve
    + [_RetrieveOp]
)

__all__ = [
    "LOGICAL_OPERATORS",
    "PHYSICAL_OPERATORS",
]
