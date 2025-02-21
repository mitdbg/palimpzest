from palimpzest.query.optimizer.rules import (
    AggregateRule as _AggregateRule,
)
from palimpzest.query.optimizer.rules import (
    BasicSubstitutionRule as _BasicSubstitutionRule,
)
from palimpzest.query.optimizer.rules import (
    CodeSynthesisConvertRule as _CodeSynthesisConvertRule,
)
from palimpzest.query.optimizer.rules import (
    CodeSynthesisConvertSingleRule as _CodeSynthesisConvertSingleRule,
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
    TokenReducedConvertBondedRule as _TokenReducedConvertBondedRule,
)
from palimpzest.query.optimizer.rules import (
    TransformationRule as _TransformationRule,
)

ALL_RULES = [
    _AggregateRule,
    _BasicSubstitutionRule,
    _CodeSynthesisConvertRule,
    _CodeSynthesisConvertSingleRule,
    _CriticAndRefineConvertRule,
    _ImplementationRule,
    _LLMConvertBondedRule,
    _LLMFilterRule,
    _MixtureOfAgentsConvertRule,
    _NonLLMConvertRule,
    _NonLLMFilterRule,
    _PushDownFilter,
    _RAGConvertRule,
    _RetrieveRule,
    _Rule,
    _TokenReducedConvertBondedRule,
    _TransformationRule,
]

IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, _ImplementationRule)
    and rule not in [_CodeSynthesisConvertRule, _ImplementationRule]
]

TRANSFORMATION_RULES = [
    rule for rule in ALL_RULES if issubclass(rule, _TransformationRule) and rule not in [_TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
