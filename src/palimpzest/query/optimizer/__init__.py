from palimpzest.query.optimizer.rules import AddContextsBeforeComputeRule as _AddContextsBeforeComputeRule
from palimpzest.query.optimizer.rules import (
    AggregateRule as _AggregateRule,
)
from palimpzest.query.optimizer.rules import (
    BasicSubstitutionRule as _BasicSubstitutionRule,
)
from palimpzest.query.optimizer.rules import (
    CriticAndRefineConvertRule as _CriticAndRefineConvertRule,
)
from palimpzest.query.optimizer.rules import (
    ImplementationRule as _ImplementationRule,
)
from palimpzest.query.optimizer.rules import (
    LLMConvertBondedRule as _LLMConvertBondedRule,
)
from palimpzest.query.optimizer.rules import (
    LLMFilterRule as _LLMFilterRule,
)
from palimpzest.query.optimizer.rules import (
    LLMJoinRule as _LLMJoinRule,
)
from palimpzest.query.optimizer.rules import (
    MixtureOfAgentsConvertRule as _MixtureOfAgentsConvertRule,
)
from palimpzest.query.optimizer.rules import (
    NonLLMConvertRule as _NonLLMConvertRule,
)
from palimpzest.query.optimizer.rules import (
    NonLLMFilterRule as _NonLLMFilterRule,
)
from palimpzest.query.optimizer.rules import (
    PushDownFilter as _PushDownFilter,
)
from palimpzest.query.optimizer.rules import (
    RAGConvertRule as _RAGConvertRule,
)
from palimpzest.query.optimizer.rules import (
    RetrieveRule as _RetrieveRule,
)
from palimpzest.query.optimizer.rules import (
    Rule as _Rule,
)
from palimpzest.query.optimizer.rules import (
    SemanticAggregateRule as _SemanticAggregateRule,
)
from palimpzest.query.optimizer.rules import (
    SplitConvertRule as _SplitConvertRule,
)
from palimpzest.query.optimizer.rules import (
    TransformationRule as _TransformationRule,
)

ALL_RULES = [
    _AddContextsBeforeComputeRule,
    _AggregateRule,
    _BasicSubstitutionRule,
    _CriticAndRefineConvertRule,
    _ImplementationRule,
    _LLMConvertBondedRule,
    _LLMFilterRule,
    _LLMJoinRule,
    _MixtureOfAgentsConvertRule,
    _NonLLMConvertRule,
    _NonLLMFilterRule,
    _PushDownFilter,
    _RAGConvertRule,
    _RetrieveRule,
    _Rule,
    _SemanticAggregateRule,
    _SplitConvertRule,
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
