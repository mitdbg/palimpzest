from palimpzest.optimizer.rules import (
    AggregateRule as _AggregateRule,
)
from palimpzest.optimizer.rules import (
    BasicSubstitutionRule as _BasicSubstitutionRule,
)
from palimpzest.optimizer.rules import (
    CodeSynthesisConvertRule as _CodeSynthesisConvertRule,
)
from palimpzest.optimizer.rules import (
    CodeSynthesisConvertSingleRule as _CodeSynthesisConvertSingleRule,
)
from palimpzest.optimizer.rules import (
    ImplementationRule as _ImplementationRule,
)
from palimpzest.optimizer.rules import (
    LLMConvertBondedRule as _LLMConvertBondedRule,
)
from palimpzest.optimizer.rules import (
    LLMConvertConventionalRule as _LLMConvertConventionalRule,
)
from palimpzest.optimizer.rules import (
    LLMConvertRule as _LLMConvertRule,
)
from palimpzest.optimizer.rules import (
    LLMFilterRule as _LLMFilterRule,
)
from palimpzest.optimizer.rules import (
    MixtureOfAgentsConvertRule as _MixtureOfAgentsConvertRule,
)
from palimpzest.optimizer.rules import (
    NonLLMConvertRule as _NonLLMConvertRule,
)
from palimpzest.optimizer.rules import (
    NonLLMFilterRule as _NonLLMFilterRule,
)
from palimpzest.optimizer.rules import (
    PushDownFilter as _PushDownFilter,
)
from palimpzest.optimizer.rules import (
    RAGConvertRule as _RAGConvertRule,
)
from palimpzest.optimizer.rules import (
    RetrieveRule as _RetrieveRule,
)
from palimpzest.optimizer.rules import (
    Rule as _Rule,
)
from palimpzest.optimizer.rules import (
    TokenReducedConvertBondedRule as _TokenReducedConvertBondedRule,
)
from palimpzest.optimizer.rules import (
    TokenReducedConvertConventionalRule as _TokenReducedConvertConventionalRule,
)
from palimpzest.optimizer.rules import (
    TokenReducedConvertRule as _TokenReducedConvertRule,
)
from palimpzest.optimizer.rules import (
    TransformationRule as _TransformationRule,
)

ALL_RULES = [
    _AggregateRule,
    _BasicSubstitutionRule,
    _CodeSynthesisConvertRule,
    _CodeSynthesisConvertSingleRule,
    _ImplementationRule,
    _LLMConvertBondedRule,
    _LLMConvertConventionalRule,
    _LLMConvertRule,
    _LLMFilterRule,
    _MixtureOfAgentsConvertRule,
    _NonLLMConvertRule,
    _NonLLMFilterRule,
    _PushDownFilter,
    _RAGConvertRule,
    _RetrieveRule,
    _Rule,
    _TokenReducedConvertBondedRule,
    _TokenReducedConvertConventionalRule,
    _TokenReducedConvertRule,
    _TransformationRule,
]

IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, _ImplementationRule)
    and rule not in [_CodeSynthesisConvertRule, _ImplementationRule, _LLMConvertRule, _RAGConvertRule, _TokenReducedConvertRule]
]

TRANSFORMATION_RULES = [
    rule for rule in ALL_RULES if issubclass(rule, _TransformationRule) and rule not in [_TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
