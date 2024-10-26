from palimpzest.optimizer.rules import (
    AggregateRule,
    BasicSubstitutionRule,
    CodeSynthesisConvertRule,
    CodeSynthesisConvertSingleRule,
    ImplementationRule,
    LLMConvertBondedRule,
    LLMConvertConventionalRule,
    LLMConvertRule,
    LLMFilterRule,
    NonLLMConvertRule,
    NonLLMFilterRule,
    PushDownFilter,
    Rule,
    TokenReducedConvertBondedRule,
    TokenReducedConvertConventionalRule,
    TokenReducedConvertRule,
    TransformationRule,
)

ALL_RULES = [
    AggregateRule,
    BasicSubstitutionRule,
    CodeSynthesisConvertRule,
    CodeSynthesisConvertSingleRule,
    ImplementationRule,
    LLMConvertBondedRule,
    LLMConvertConventionalRule,
    LLMConvertRule,
    LLMFilterRule,
    NonLLMConvertRule,
    NonLLMFilterRule,
    PushDownFilter,
    Rule,
    TokenReducedConvertBondedRule,
    TokenReducedConvertConventionalRule,
    TokenReducedConvertRule,
    TransformationRule,
]
IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, ImplementationRule)
    and rule not in [ImplementationRule, LLMConvertRule, TokenReducedConvertRule, CodeSynthesisConvertRule]
]
TRANSFORMATION_RULES = [
    rule for rule in ALL_RULES if issubclass(rule, TransformationRule) and rule not in [TransformationRule]
]

__all__ = [
    "ALL_RULES",
    "IMPLEMENTATION_RULES",
    "TRANSFORMATION_RULES",
]
