from palimpzest.query.operators.aggregate import AggregateOp as _AggregateOp
from palimpzest.query.operators.aggregate import ApplyGroupByOp as _ApplyGroupByOp
from palimpzest.query.operators.aggregate import AverageAggregateOp as _AverageAggregateOp
from palimpzest.query.operators.aggregate import CountAggregateOp as _CountAggregateOp
from palimpzest.query.operators.aggregate import MaxAggregateOp as _MaxAggregateOp
from palimpzest.query.operators.aggregate import MinAggregateOp as _MinAggregateOp
from palimpzest.query.operators.aggregate import SemanticAggregate as _SemanticAggregate
from palimpzest.query.operators.aggregate import SumAggregateOp as _SumAggregateOp
from palimpzest.query.operators.convert import ConvertOp as _ConvertOp
from palimpzest.query.operators.convert import LLMConvert as _LLMConvert
from palimpzest.query.operators.convert import LLMConvertBonded as _LLMConvertBonded
from palimpzest.query.operators.convert import NonLLMConvert as _NonLLMConvert
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineConvert as _CritiqueAndRefineConvert
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineFilter as _CritiqueAndRefineFilter
from palimpzest.query.operators.distinct import DistinctOp as _DistinctOp
from palimpzest.query.operators.filter import FilterOp as _FilterOp
from palimpzest.query.operators.filter import LLMFilter as _LLMFilter
from palimpzest.query.operators.filter import NonLLMFilter as _NonLLMFilter
from palimpzest.query.operators.join import EmbeddingJoin as _EmbeddingJoin
from palimpzest.query.operators.join import JoinOp as _JoinOp
from palimpzest.query.operators.join import NestedLoopsJoin as _NestedLoopsJoin
from palimpzest.query.operators.limit import LimitScanOp as _LimitScanOp
from palimpzest.query.operators.logical import (
    Aggregate as _Aggregate,
)
from palimpzest.query.operators.logical import (
    BaseScan as _BaseScan,
)
from palimpzest.query.operators.logical import (
    ConvertScan as _ConvertScan,
)
from palimpzest.query.operators.logical import (
    Distinct as _Distinct,
)
from palimpzest.query.operators.logical import (
    FilteredScan as _FilteredScan,
)
from palimpzest.query.operators.logical import (
    GroupByAggregate as _GroupByAggregate,
)
from palimpzest.query.operators.logical import (
    JoinOp as _LogicalJoinOp,
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
    TopKScan as _TopKScan,
)
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsConvert as _MixtureOfAgentsConvert
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsFilter as _MixtureOfAgentsFilter
from palimpzest.query.operators.physical import PhysicalOperator as _PhysicalOperator
from palimpzest.query.operators.project import ProjectOp as _ProjectOp
from palimpzest.query.operators.rag import RAGConvert as _RAGConvert
from palimpzest.query.operators.rag import RAGFilter as _RAGFilter
from palimpzest.query.operators.scan import MarshalAndScanDataOp as _MarshalAndScanDataOp
from palimpzest.query.operators.scan import ScanPhysicalOp as _ScanPhysicalOp
from palimpzest.query.operators.split import SplitConvert as _SplitConvert
from palimpzest.query.operators.split import SplitFilter as _SplitFilter
from palimpzest.query.operators.topk import TopKOp as _TopKOp

LOGICAL_OPERATORS = [
    _LogicalOperator,
    _Aggregate,
    _BaseScan,
    _ConvertScan,
    _Distinct,
    _FilteredScan,
    _GroupByAggregate,
    _LogicalJoinOp,
    _LimitScan,
    _Project,
    _TopKScan,
]

PHYSICAL_OPERATORS = (
    # aggregate
    [_AggregateOp, _ApplyGroupByOp, _AverageAggregateOp, _CountAggregateOp, _MaxAggregateOp, _MinAggregateOp, _SemanticAggregate, _SumAggregateOp]
    # convert
    + [_ConvertOp, _NonLLMConvert, _LLMConvert, _LLMConvertBonded]
    # critique and refine
    + [_CritiqueAndRefineConvert, _CritiqueAndRefineFilter]
    # distinct
    + [_DistinctOp]
    # scan
    + [_ScanPhysicalOp, _MarshalAndScanDataOp]
    # filter
    + [_FilterOp, _NonLLMFilter, _LLMFilter]
    # join
    + [_EmbeddingJoin, _JoinOp, _NestedLoopsJoin]
    # limit
    + [_LimitScanOp]
    # mixture-of-agents
    + [_MixtureOfAgentsConvert, _MixtureOfAgentsFilter]
    # physical
    + [_PhysicalOperator]
    # project
    + [_ProjectOp]
    # rag
    + [_RAGConvert, _RAGFilter]
    # top-k
    + [_TopKOp]
    # split
    + [_SplitConvert, _SplitFilter]
)

__all__ = [
    "LOGICAL_OPERATORS",
    "PHYSICAL_OPERATORS",
]
