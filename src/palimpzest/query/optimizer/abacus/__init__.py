from palimpzest.query.optimizer.abacus.rules import AddContextsBeforeComputeRule as _AddContextsBeforeComputeRule
from palimpzest.query.optimizer.abacus.rules import (
    AggregateRule as _AggregateRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    BatchedFilterRule as _BatchedFilterRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    BasicSubstitutionRule as _BasicSubstitutionRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    CritiqueAndRefineRule as _CritiqueAndRefineRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    EmbeddingJoinRule as _EmbeddingJoinRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    ImplementationRule as _ImplementationRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    LLMConvertBondedRule as _LLMConvertBondedRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    LLMFilterRule as _LLMFilterRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    MixtureOfAgentsRule as _MixtureOfAgentsRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    NestedLoopsJoinRule as _NestedLoopsJoinRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    NonLLMConvertRule as _NonLLMConvertRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    NonLLMFilterRule as _NonLLMFilterRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    PushDownFilter as _PushDownFilter,
)
from palimpzest.query.optimizer.abacus.rules import (
    RAGRule as _RAGRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    RelationalJoinRule as _RelationalJoinRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    ReorderConverts as _ReorderConverts,
)
from palimpzest.query.optimizer.abacus.rules import (
    RescaledImageFilterRule as _RescaledImageFilterRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    Rule as _Rule,
)
from palimpzest.query.optimizer.abacus.rules import (
    SemanticAggregateRule as _SemanticAggregateRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    SplitRule as _SplitRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    TopKRule as _TopKRule,
)
from palimpzest.query.optimizer.abacus.rules import (
    TransformationRule as _TransformationRule,
)


ALL_RULES = [
    _AddContextsBeforeComputeRule,
    _AggregateRule,
    _BasicSubstitutionRule,
    _BatchedFilterRule,
    _CritiqueAndRefineRule,
    _EmbeddingJoinRule,
    _ImplementationRule,
    _LLMConvertBondedRule,
    _LLMFilterRule,
    _NestedLoopsJoinRule,
    _MixtureOfAgentsRule,
    _NonLLMConvertRule,
    _NonLLMFilterRule,
    _PushDownFilter,
    _RAGRule,
    _RelationalJoinRule,
    _ReorderConverts,
    _RescaledImageFilterRule,
    _TopKRule,
    _Rule,
    _SemanticAggregateRule,
    _SplitRule,
    _TransformationRule,
]

IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, _ImplementationRule)
    and rule not in [_ImplementationRule]
]

TRANSFORMATION_RULES = [
    rule for rule in ALL_RULES if issubclass(rule, _TransformationRule) and rule not in [_TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
